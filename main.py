#!/usr/bin/env python3

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')
from gi.repository import Gst, GstApp, GObject
import sys
import pyds
import json
import datetime
import threading
import grpc
from concurrent import futures

# Import local modules
from app.pipeline_builder import create_pipeline
from app.grpc_server import FrameStreamerServicer, start_grpc_server
from app.grpc_client import MetadataGrpcClient
from app.custom_probes import all_data_probe

def main():
    # Initialize GStreamer
    Gst.init(None)
    
    # Create main loop
    loop = GObject.MainLoop()
    
    # Configuration
    config = {
        'input_type': 'rtsp',  # or 'grpc'
        'rtsp_uri': 'rtsp://your_camera_url',  # if using RTSP
        'grpc_server_port': 50051,  # if using gRPC input
        'grpc_client_host': 'localhost',
        'grpc_client_port': 50052,
        'display': False  # Set to True for local display
    }
    
    # Create gRPC client for sending metadata
    grpc_client = MetadataGrpcClient(
        host=config['grpc_client_host'],
        port=config['grpc_client_port']
    )
    
    # Create pipeline
    pipeline = create_pipeline(config, grpc_client)
    
    # Set pipeline state to playing
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("Failed to start pipeline")
        sys.exit(1)
    
    # Start gRPC server if using gRPC input
    if config['input_type'] == 'grpc':
        appsrc = pipeline.get_by_name('appsrc')
        if not appsrc:
            print("Failed to get appsrc element")
            sys.exit(1)
        
        server_thread = threading.Thread(
            target=start_grpc_server,
            args=(appsrc, config['grpc_server_port'])
        )
        server_thread.daemon = True
        server_thread.start()
    
    try:
        # Run main loop
        loop.run()
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        pipeline.set_state(Gst.State.NULL)
        loop.quit()

if __name__ == '__main__':
    main() 