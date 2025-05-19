#!/usr/bin/env python3

import grpc
from concurrent import futures
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from generated_pb2 import deepstream_grpc_pb2, deepstream_grpc_pb2_grpc
import threading
import queue
import numpy as np

class FrameStreamerServicer(deepstream_grpc_pb2_grpc.FrameStreamerServicer):
    def __init__(self, appsrc_element):
        """Initialize the frame streamer servicer.
        
        Args:
            appsrc_element: GStreamer appsrc element to push frames to
        """
        self.appsrc = appsrc_element
        self.frame_queue = queue.Queue(maxsize=30)  # Buffer up to 30 frames
        self.running = True
        
        # Start frame processing thread
        self.process_thread = threading.Thread(target=self._process_frames)
        self.process_thread.daemon = True
        self.process_thread.start()
    
    def StreamFrames(self, request_iterator, context):
        """Stream frames from the client to the appsrc element.
        
        Args:
            request_iterator: Iterator of VideoFrame messages
            context: gRPC context
            
        Returns:
            StreamAck message
        """
        try:
            for frame in request_iterator:
                # Put frame in queue for processing
                self.frame_queue.put(frame, block=True, timeout=1.0)
            
            return deepstream_grpc_pb2.StreamAck(
                success=True,
                message="Stream completed successfully"
            )
            
        except Exception as e:
            print(f"Error in StreamFrames: {e}")
            return deepstream_grpc_pb2.StreamAck(
                success=False,
                message=str(e)
            )
    
    def _process_frames(self):
        """Process frames from the queue and push them to appsrc."""
        while self.running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=1.0)
                
                # Convert frame data to numpy array
                frame_data = np.frombuffer(frame.frame_data, dtype=np.uint8)
                frame_data = frame_data.reshape((frame.height, frame.width, 3))
                
                # Create GStreamer buffer
                buffer = Gst.Buffer.new_wrapped(frame_data.tobytes())
                
                # Set timestamp
                buffer.pts = frame.timestamp_us * 1000  # Convert to nanoseconds
                buffer.dts = Gst.CLOCK_TIME_NONE
                buffer.duration = Gst.CLOCK_TIME_NONE
                
                # Push buffer to appsrc
                ret = self.appsrc.emit("push-buffer", buffer)
                if ret != Gst.FlowReturn.OK:
                    print(f"Error pushing buffer: {ret}")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing frame: {e}")
    
    def stop(self):
        """Stop the frame processing thread."""
        self.running = False
        if self.process_thread.is_alive():
            self.process_thread.join()

def start_grpc_server(appsrc_element, port: int = 50051):
    """Start the gRPC server.
    
    Args:
        appsrc_element: GStreamer appsrc element
        port: Port number to listen on
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = FrameStreamerServicer(appsrc_element)
    deepstream_grpc_pb2_grpc.add_FrameStreamerServicer_to_server(
        servicer, server
    )
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        servicer.stop()
        server.stop(0) 