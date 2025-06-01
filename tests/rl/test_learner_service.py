# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Event, Queue
from typing import Generator

import grpc
import pytest
import torch

from lerobot.common.transport import services_pb2, services_pb2_grpc
from lerobot.common.transport.utils import (
    bytes_to_python_object,
    bytes_to_state_dict,
    bytes_to_transitions,
    python_object_to_bytes,
    state_to_bytes,
    transitions_to_bytes,
)
from lerobot.common.utils.transition import Transition
from lerobot.scripts.rl.learner_service import SHUTDOWN_TIMEOUT, LearnerService


class TestLearnerService:
    """Test suite for LearnerService gRPC functionality."""

    @pytest.fixture
    def grpc_server_port(self) -> int:
        """Return a free port for gRPC server."""
        return 50051

    @pytest.fixture
    def shutdown_event(self) -> Event:
        """Create a shutdown event for the service."""
        return Event()

    @pytest.fixture
    def parameters_queue(self) -> Queue:
        """Create a queue for parameters."""
        return Queue()

    @pytest.fixture
    def transition_queue(self) -> Queue:
        """Create a queue for transitions."""
        return Queue()

    @pytest.fixture
    def interaction_queue(self) -> Queue:
        """Create a queue for interactions."""
        return Queue()

    @pytest.fixture
    def learner_service(
        self,
        shutdown_event: Event,
        parameters_queue: Queue,
        transition_queue: Queue,
        interaction_queue: Queue,
    ) -> LearnerService:
        """Create a LearnerService instance."""
        return LearnerService(
            shutdown_event=shutdown_event,
            parameters_queue=parameters_queue,
            seconds_between_pushes=0.1,  # Fast for testing
            transition_queue=transition_queue,
            interaction_message_queue=interaction_queue,
        )

    @pytest.fixture
    def grpc_server(
        self, learner_service: LearnerService, grpc_server_port: int
    ) -> Generator[grpc.Server, None, None]:
        """Create and start a gRPC server with the learner service."""
        server = grpc.server(ThreadPoolExecutor(max_workers=10))
        services_pb2_grpc.add_LearnerServiceServicer_to_server(learner_service, server)

        listen_addr = f"localhost:{grpc_server_port}"
        server.add_insecure_port(listen_addr)
        server.start()

        yield server

        server.stop(SHUTDOWN_TIMEOUT)

    @pytest.fixture
    def grpc_channel(
        self, grpc_server: grpc.Server, grpc_server_port: int
    ) -> Generator[grpc.Channel, None, None]:
        """Create a gRPC channel to connect to the test server."""
        channel = grpc.insecure_channel(f"localhost:{grpc_server_port}")
        yield channel
        channel.close()

    @pytest.fixture
    def grpc_stub(self, grpc_channel: grpc.Channel) -> services_pb2_grpc.LearnerServiceStub:
        """Create a gRPC stub for making requests."""
        return services_pb2_grpc.LearnerServiceStub(grpc_channel)

    def test_ready_endpoint(self, grpc_stub: services_pb2_grpc.LearnerServiceStub):
        """Test the Ready endpoint returns successfully."""
        response = grpc_stub.Ready(services_pb2.Empty())
        assert isinstance(response, services_pb2.Empty)

    def test_stream_parameters_single_parameter(
        self,
        grpc_stub: services_pb2_grpc.LearnerServiceStub,
        parameters_queue: Queue,
        shutdown_event: Event,
    ):
        """Test streaming a single parameter set."""
        # Create test model parameters
        test_params = {
            "layer1.weight": torch.randn(10, 5),
            "layer1.bias": torch.randn(10),
            "layer2.weight": torch.randn(1, 10),
            "layer2.bias": torch.randn(1),
        }

        # Serialize parameters
        params_bytes = state_to_bytes(test_params)
        parameters_queue.put(params_bytes)

        # Start streaming in a separate thread
        def shutdown_after_delay():
            time.sleep(0.5)  # Let one parameter stream
            shutdown_event.set()

        shutdown_thread = threading.Thread(target=shutdown_after_delay)
        shutdown_thread.start()

        # Collect streamed parameters
        received_chunks = []
        for response in grpc_stub.StreamParameters(services_pb2.Empty()):
            received_chunks.append(response)

        shutdown_thread.join()

        # Verify we received data
        assert len(received_chunks) > 0

        # Reconstruct the full message
        full_data = b""
        for chunk in received_chunks:
            full_data += chunk.data

        # Verify the reconstructed parameters match
        reconstructed_params = bytes_to_state_dict(full_data)

        assert len(reconstructed_params) == len(test_params)
        for key in test_params:
            assert key in reconstructed_params
            assert torch.allclose(test_params[key], reconstructed_params[key])

    def test_send_transitions(
        self,
        grpc_stub: services_pb2_grpc.LearnerServiceStub,
        transition_queue: Queue,
        shutdown_event: Event,
    ):
        """Test sending transitions to the learner."""
        # Create test transitions
        test_transitions = [
            Transition(
                observation={"image": torch.randn(3, 64, 64), "state": torch.randn(10)},
                action=torch.randn(5),
                reward=torch.tensor(1.0),
                done=torch.tensor(False),
                next_observation={"image": torch.randn(3, 64, 64), "state": torch.randn(10)},
            ),
            Transition(
                observation={"image": torch.randn(3, 64, 64), "state": torch.randn(10)},
                action=torch.randn(5),
                reward=torch.tensor(-0.1),
                done=torch.tensor(True),
                next_observation={"image": torch.randn(3, 64, 64), "state": torch.randn(10)},
            ),
        ]

        # Serialize transitions
        transitions_bytes = transitions_to_bytes(test_transitions)

        # Create chunks for streaming
        def create_transition_chunks():
            chunk_size = 1024  # Small chunks for testing
            data = transitions_bytes

            for i in range(0, len(data), chunk_size):
                if i == 0:
                    transfer_state = services_pb2.TransferState.TRANSFER_BEGIN
                elif i + chunk_size >= len(data):
                    transfer_state = services_pb2.TransferState.TRANSFER_END
                else:
                    transfer_state = services_pb2.TransferState.TRANSFER_MIDDLE

                chunk = data[i : i + chunk_size]
                yield services_pb2.Transition(transfer_state=transfer_state, data=chunk)

        # Send transitions
        response = grpc_stub.SendTransitions(create_transition_chunks())
        assert isinstance(response, services_pb2.Empty)

        # Wait for processing
        time.sleep(0.1)

        # Verify transitions were received
        assert not transition_queue.empty()
        received_bytes = transition_queue.get()
        received_transitions = bytes_to_transitions(received_bytes)

        assert len(received_transitions) == len(test_transitions)

        # Verify transition content
        for original, received in zip(test_transitions, received_transitions, strict=False):
            assert torch.allclose(original.action, received.action)
            assert torch.allclose(original.reward, received.reward)
            assert torch.equal(original.done, received.done)

    def test_send_interactions(
        self,
        grpc_stub: services_pb2_grpc.LearnerServiceStub,
        interaction_queue: Queue,
        shutdown_event: Event,
    ):
        """Test sending interaction messages to the learner."""
        # Create test interaction data
        test_interaction = {
            "episode_id": 12345,
            "step": 42,
            "metadata": {"env_id": "test_env", "timestamp": 1640995200.0},
            "custom_data": torch.randn(5, 10),
        }

        # Serialize interaction
        interaction_bytes = python_object_to_bytes(test_interaction)

        # Create chunks for streaming
        def create_interaction_chunks():
            chunk_size = 512  # Small chunks for testing
            data = interaction_bytes

            for i in range(0, len(data), chunk_size):
                if i == 0:
                    transfer_state = services_pb2.TransferState.TRANSFER_BEGIN
                elif i + chunk_size >= len(data):
                    transfer_state = services_pb2.TransferState.TRANSFER_END
                else:
                    transfer_state = services_pb2.TransferState.TRANSFER_MIDDLE

                chunk = data[i : i + chunk_size]
                yield services_pb2.InteractionMessage(transfer_state=transfer_state, data=chunk)

        # Send interactions
        response = grpc_stub.SendInteractions(create_interaction_chunks())
        assert isinstance(response, services_pb2.Empty)

        # Wait for processing
        time.sleep(0.1)

        # Verify interactions were received
        assert not interaction_queue.empty()
        received_bytes = interaction_queue.get()
        received_interaction = bytes_to_python_object(received_bytes)

        # Verify interaction content
        assert received_interaction["episode_id"] == test_interaction["episode_id"]
        assert received_interaction["step"] == test_interaction["step"]
        assert received_interaction["metadata"] == test_interaction["metadata"]
        assert torch.allclose(received_interaction["custom_data"], test_interaction["custom_data"])

    def test_concurrent_operations(
        self,
        grpc_stub: services_pb2_grpc.LearnerServiceStub,
        parameters_queue: Queue,
        transition_queue: Queue,
        interaction_queue: Queue,
        shutdown_event: Event,
    ):
        """Test that multiple operations can run concurrently."""
        # Setup test data
        test_params = {"test_param": torch.randn(5, 5)}
        params_bytes = state_to_bytes(test_params)
        parameters_queue.put(params_bytes)

        test_transitions = [
            Transition(
                observation={"state": torch.randn(3)},
                action=torch.randn(2),
                reward=torch.tensor(0.5),
                done=torch.tensor(False),
                next_observation={"state": torch.randn(3)},
            )
        ]
        transitions_bytes = transitions_to_bytes(test_transitions)

        test_interaction = {"test": "data"}
        interaction_bytes = python_object_to_bytes(test_interaction)

        # Helper functions for concurrent operations
        def stream_parameters():
            chunks = list(grpc_stub.StreamParameters(services_pb2.Empty()))
            return len(chunks)

        def send_transitions():
            chunks = [
                services_pb2.Transition(
                    transfer_state=services_pb2.TransferState.TRANSFER_END, data=transitions_bytes
                )
            ]
            response = grpc_stub.SendTransitions(iter(chunks))
            return isinstance(response, services_pb2.Empty)

        def send_interactions():
            chunks = [
                services_pb2.InteractionMessage(
                    transfer_state=services_pb2.TransferState.TRANSFER_END, data=interaction_bytes
                )
            ]
            response = grpc_stub.SendInteractions(iter(chunks))
            return isinstance(response, services_pb2.Empty)

        # Run operations concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Start parameter streaming
            param_future = executor.submit(stream_parameters)

            # Small delay to ensure streaming starts
            time.sleep(0.1)

            # Send transitions and interactions concurrently
            transition_future = executor.submit(send_transitions)
            interaction_future = executor.submit(send_interactions)

            # Wait for sends to complete
            assert transition_future.result()
            assert interaction_future.result()

            # Stop parameter streaming
            shutdown_event.set()
            param_count = param_future.result()

        # Verify all operations completed successfully
        assert param_count > 0

        # Check queues received data
        time.sleep(0.1)
        assert not transition_queue.empty()
        assert not interaction_queue.empty()

    def test_error_handling_invalid_data(
        self,
        grpc_stub: services_pb2_grpc.LearnerServiceStub,
    ):
        """Test error handling with invalid/corrupted data."""

        # Send corrupted transition data
        def create_bad_chunks():
            yield services_pb2.Transition(
                transfer_state=services_pb2.TransferState.TRANSFER_END, data=b"invalid_pickle_data"
            )

        # This should not crash the server
        try:
            response = grpc_stub.SendTransitions(create_bad_chunks())
            assert isinstance(response, services_pb2.Empty)
        except grpc.RpcError as e:
            # Some gRPC errors might be expected with corrupted data
            assert e.code() in [grpc.StatusCode.INVALID_ARGUMENT, grpc.StatusCode.INTERNAL]

    def test_large_data_transfer(
        self,
        grpc_stub: services_pb2_grpc.LearnerServiceStub,
        parameters_queue: Queue,
        shutdown_event: Event,
    ):
        """Test handling of large data transfers (multiple chunks)."""
        # Create large parameter tensor
        large_params = {
            "large_layer": torch.randn(1000, 1000),  # ~4MB tensor
            "small_layer": torch.randn(10),
        }

        params_bytes = state_to_bytes(large_params)
        parameters_queue.put(params_bytes)

        # Stream parameters and collect all chunks
        def collect_parameters():
            chunks = []
            for chunk in grpc_stub.StreamParameters(services_pb2.Empty()):
                chunks.append(chunk)
                # Stop after collecting some chunks
                if len(chunks) >= 3:
                    break
            return chunks

        # Start collection in thread
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(collect_parameters)

            # Let it collect some chunks then shutdown
            time.sleep(0.2)
            shutdown_event.set()

            chunks = future.result()

        # Verify we received multiple chunks for large data
        assert len(chunks) >= 2

        # Verify chunks have proper transfer states
        if len(chunks) > 1:
            assert chunks[0].transfer_state == services_pb2.TransferState.TRANSFER_BEGIN
            for chunk in chunks[1:-1]:
                assert chunk.transfer_state == services_pb2.TransferState.TRANSFER_MIDDLE


if __name__ == "__main__":
    # Enable logging for debugging
    logging.basicConfig(level=logging.INFO)

    # Run tests with pytest
    pytest.main([__file__, "-v"])
