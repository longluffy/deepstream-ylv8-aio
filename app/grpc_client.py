#!/usr/bin/env python3

import grpc
import json
from generated_pb2 import deepstream_grpc_pb2, deepstream_grpc_pb2_grpc
import time

class MetadataGrpcClient:
    def __init__(self, host: str = 'localhost', port: int = 50052):
        """Initialize the gRPC client.
        
        Args:
            host: Host address of the gRPC server
            port: Port number of the gRPC server
        """
        self.channel = grpc.insecure_channel(f'{host}:{port}')
        self.stub = deepstream_grpc_pb2_grpc.ResultReceiverStub(self.channel)
    
    def send_metadata(self, frame_json: dict) -> bool:
        """Send frame metadata to the NX Optix plugin.
        
        Args:
            frame_json: Dictionary containing frame metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert dictionary to JSON string
            json_str = json.dumps(frame_json)
            
            # Create ResultData message
            result = deepstream_grpc_pb2.ResultData(
                json_payload=json_str,
                timestamp_us=int(time.time() * 1e6),
                source_id=frame_json.get('source_id', 'unknown')
            )
            
            # Send result
            response = self.stub.SendResult(result)
            return response.success
            
        except grpc.RpcError as e:
            print(f"gRPC error: {e}")
            return False
        except Exception as e:
            print(f"Error sending metadata: {e}")
            return False
    
    def close(self):
        """Close the gRPC channel."""
        if self.channel:
            self.channel.close() 