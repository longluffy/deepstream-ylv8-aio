#!/usr/bin/env python3

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import pyds
import json
import datetime
import numpy as np
from app.identity_manager import IdentityManager

# Global dictionary to store known face embeddings
known_face_embeddings = {}
identity_manager = IdentityManager()

def all_data_probe(pad, info, u_data):
    """Probe function that processes all metadata and sends it via gRPC."""
    grpc_client = u_data
    buffer = info.get_buffer()
    if not buffer:
        return Gst.PadProbeReturn.OK
    
    # Get batch metadata
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
    
    # Process each frame in the batch
    for frame_meta in pyds.NvDsFrameMetaList(batch_meta.frame_meta_list):
        frame_objects = []
        
        # Process each object in the frame
        for obj_meta in pyds.NvDsObjectMetaList(frame_meta.obj_meta_list):
            # Only process person detections
            if obj_meta.class_id != 0:  # Assuming 0 is person class
                continue
            
            # Get tracking ID
            track_id = obj_meta.object_id
            
            # Get person bounding box
            rect = obj_meta.rect_params
            person_bbox = [rect.left, rect.top, rect.width, rect.height]
            
            # Check if we have face embedding for this track ID
            face_embedding = None
            face_identity = "unknown"
            
            # Look for face embedding in user metadata
            for user_meta in pyds.NvDsUserMetaList(obj_meta.obj_user_meta_list):
                if user_meta.base_meta.meta_type == pyds.NVDSINFER_TENSOR_OUTPUT_META:
                    tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)
                    # Get embedding from tensor output
                    face_embedding = pyds.get_infer_output_tensor(tensor_meta, 0)
                    
                    # If we have a new embedding, store it and try to identify
                    if face_embedding is not None and track_id not in known_face_embeddings:
                        known_face_embeddings[track_id] = {
                            "embedding": face_embedding,
                            "last_seen_frame": frame_meta.frame_num
                        }
                        # Try to identify the face
                        face_identity = identity_manager.identify_face(face_embedding)
                    elif track_id in known_face_embeddings:
                        face_identity = known_face_embeddings[track_id].get("identity", "unknown")
            
            # Create object data
            obj_data = {
                "track_id": int(track_id),
                "class": "person",
                "person_bbox": person_bbox,
                "face_identity": face_identity,
                "face_embedding_available": face_embedding is not None,
                "confidence_person": float(obj_meta.confidence)
            }
            
            frame_objects.append(obj_data)
        
        # Create frame-level JSON
        frame_json = {
            "timestamp": str(datetime.datetime.now(datetime.timezone.utc).isoformat()),
            "frame_id": int(frame_meta.frame_num),
            "source_id": pyds.get_string(frame_meta.source_id),
            "objects": frame_objects
        }
        
        # Send metadata via gRPC
        try:
            grpc_client.send_metadata(frame_json)
        except Exception as e:
            print(f"Error sending metadata via gRPC: {e}")
    
    return Gst.PadProbeReturn.OK 