#!/usr/bin/env python3

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject
import pyds
from app.custom_probes import all_data_probe

def create_pipeline(config, grpc_client):
    """Create and configure the DeepStream pipeline."""
    # Create pipeline
    pipeline = Gst.Pipeline.new("deepstream-pipeline")
    
    # Create elements
    if config['input_type'] == 'rtsp':
        source = Gst.ElementFactory.make("uridecodebin", "source")
        source.set_property('uri', config['rtsp_uri'])
    else:  # gRPC input
        source = Gst.ElementFactory.make("appsrc", "appsrc")
        source.set_property('format', Gst.Format.TIME)
        source.set_property('is-live', True)
        source.set_property('do-timestamp', True)
        caps = Gst.Caps.from_string(
            "video/x-raw,format=RGB,width=1920,height=1080,framerate=30/1"
        )
        source.set_property('caps', caps)
    
    # Create streammux
    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 40000)
    streammux.set_property('live-source', 1)
    
    # Create primary inference (YOLOv8)
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property('config-file-path', 'configs/config_infer_primary_yolo.txt')
    
    # Create tracker
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    tracker.set_property('ll-lib-file', '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so')
    tracker.set_property('ll-config-file', 'configs/tracker_config.yml')
    tracker.set_property('gpu-id', 0)
    
    # Create secondary inference (ArcFace)
    sgie = Gst.ElementFactory.make("nvinfer", "secondary-inference-arcface")
    sgie.set_property('config-file-path', 'configs/config_infer_secondary_arcface.txt')
    
    # Create display elements if needed
    if config['display']:
        videoconvert = Gst.ElementFactory.make("nvvideoconvert", "converter")
        osd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    else:
        sink = Gst.ElementFactory.make("fakesink", "fakesink")
    
    # Add elements to pipeline
    elements = [source, streammux, pgie, tracker, sgie]
    if config['display']:
        elements.extend([videoconvert, osd])
    elements.append(sink)
    
    for element in elements:
        if not element:
            print(f"Failed to create {element}")
            return None
        pipeline.add(element)
    
    # Link elements
    if config['input_type'] == 'rtsp':
        # Connect pad-added signal for uridecodebin
        source.connect("pad-added", cb_newpad, streammux)
    else:
        # Link appsrc to streammux
        if not source.link(streammux):
            print("Failed to link source to streammux")
            return None
    
    # Link remaining elements
    if not streammux.link(pgie):
        print("Failed to link streammux to pgie")
        return None
    
    if not pgie.link(tracker):
        print("Failed to link pgie to tracker")
        return None
    
    if not tracker.link(sgie):
        print("Failed to link tracker to sgie")
        return None
    
    if config['display']:
        if not sgie.link(videoconvert):
            print("Failed to link sgie to videoconvert")
            return None
        if not videoconvert.link(osd):
            print("Failed to link videoconvert to osd")
            return None
        if not osd.link(sink):
            print("Failed to link osd to sink")
            return None
    else:
        if not sgie.link(sink):
            print("Failed to link sgie to sink")
            return None
    
    # Add probe to sgie source pad
    sgie_src_pad = sgie.get_static_pad("src")
    sgie_src_pad.add_probe(
        Gst.PadProbeType.BUFFER,
        all_data_probe,
        grpc_client
    )
    
    return pipeline

def cb_newpad(decodebin, decoder_src_pad, data):
    """Callback for pad-added signal from uridecodebin."""
    streammux = data
    caps = decoder_src_pad.query_caps(None)
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    
    # Check if the pad is video
    if gstname.find("video") != -1:
        # Link the decodebin pad to streammux sink pad
        streammux_sink_pad = streammux.get_request_pad("sink_0")
        if not streammux_sink_pad:
            print("Failed to get streammux sink pad")
            return
        
        if not decoder_src_pad.link(streammux_sink_pad):
            print("Failed to link decoder src pad to streammux sink pad")
            return 