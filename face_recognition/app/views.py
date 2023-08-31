from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse,StreamingHttpResponse
from app.forms import SearchForm
from app.machinel_learning import pipeline_openvino
from django.conf import settings
from app.models import FaceDB
import os
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from django.core.files.uploadedfile import InMemoryUploadedFile
from django.views.decorators import gzip
BASE_PATH = '/home/eduardolacava/open_model_zoo/demos'
import sys
sys.path.insert(0,BASE_PATH + "/common/python")
sys.path.insert(0,BASE_PATH + "/common/python/openvino/model_zoo")
sys.path.insert(0,'/home/eduardolacava/fr_project/face_recognition/app')
import logging as log
import sys
from pathlib import Path
from openvino.runtime import Core, get_version

from landmarks_detector import LandmarksDetector
from utils import crop
#import monitorscomm 
from helpers import resolution
from images_capture import open_images_capture
from face_detector import FaceDetector
from faces_database import FacesDatabase
from face_identifier import FaceIdentifier
from model_api.models import OutputTransform
from model_api.performance_metrics import PerformanceMetrics

# Create your views here.

# Função para capturar frames da câmera
def generate_frames(camera):

    class FrameProcessor:

        # Path to an .xml file with Face Detection model
        m_fd = f'{BASE_PATH}/face_recognition_demo/python/intel/face-detection-retail-0004/FP32/face-detection-retail-0004.xml' 
        
        #  Target device for Face Detection model. default CPU
        d_fd = 'CPU'

        # Optional. Target device for Facial Landmarks Detection. default CPU

        d_lm  = 'CPU'

        # Optional. Target device for Face Reidentification. default CPU

        d_reid = 'CPU'
        
        # Specify the input size of detection model for reshaping. Example: 500 700
        fd_input_size = (0, 0)    
        
        # Optional. Probability threshold for face detections. default = 0.3
        t_fd = 0.3         

        # Optional. Scaling ratio for bboxes passed to face recognition. default = 1.15
        exp_r_fd = 1.15          

        # Path to an .xml file with Facial Landmarks Detection model
        m_lm =   f'{BASE_PATH}/face_recognition_demo/python/intel/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml'          

        # Required. Path to an .xml file with Face Reidentification model
        m_reid =   f'{BASE_PATH}/face_recognition_demo/python/intel/face-reidentification-retail-0095/FP32/face-reidentification-retail-0095.xml'

        # Cosine distance threshold between two vectors for face identification default = 0.3  
        t_id = 0.3        
        
        #Optional. Path to the face images directory . Default ''
        fg = '/home/eduardolacava/faces/face'

        # Optional. Algorithm for face matching. Default: HUNGARIAN
        match_algo = 'HUNGARIAN' 

        #Optional. Don't show output
        no_show = True

        # Optional. Use Face Detection model to find faces
        run_detector = False


        QUEUE_SIZE = 16

        def __init__(self):
            
            log.info('OpenVINO Runtime')
            log.info('\tbuild: {}'.format(get_version()))
            core = Core()

            self.face_detector = FaceDetector(core, self.m_fd,
                                            self.fd_input_size,
                                            confidence_threshold=self.t_fd,
                                            roi_scale_factor=self.exp_r_fd)
            self.landmarks_detector = LandmarksDetector(core, self.m_lm)
            self.face_identifier = FaceIdentifier(core, self.m_reid,
                                                match_threshold=self.t_id,
                                                match_algo=self.match_algo)

            self.face_detector.deploy(self.d_fd)
            self.landmarks_detector.deploy(self.d_lm, self.QUEUE_SIZE)
            self.face_identifier.deploy(self.d_reid, self.QUEUE_SIZE)

            log.debug('Building faces database using images from {}'.format(self.fg))
            self.faces_database = FacesDatabase(self.fg, self.face_identifier,
                                                self.landmarks_detector,
                                                self.face_detector if self.run_detector else None, self.no_show)
            self.face_identifier.set_faces_database(self.faces_database)
            log.info('Database is built, registered {} identities'.format(len(self.faces_database)))

        def process(self, frame):
                orig_image = frame.copy()

                rois = self.face_detector.infer((frame,))
                if self.QUEUE_SIZE < len(rois):
                    log.warning('Too many faces for processing. Will be processed only {} of {}'
                                .format(self.QUEUE_SIZE, len(rois)))
                    rois = rois[:self.QUEUE_SIZE]

                landmarks = self.landmarks_detector.infer((frame, rois))
                face_identities, unknowns = self.face_identifier.infer((frame, rois, landmarks))

                return [rois, landmarks, face_identities]
   

    frame_processor = FrameProcessor()
    while True:
        # Captura um frame da câmera
        success, frame = camera.read()

        if not success:
            break
        else:
            # Converte o frame em um formato que possa ser exibido em HTML
            detections,frame,detects  = pipeline_openvino(frame,
                                                          frame_processor)



            _, buffer = cv2.imencode('.jpg', frame)

            frame_data = buffer.tobytes()
            
            
            if detections:
                for roi, landmarks, identity in zip(*detects):
                    name = frame_processor.face_identifier.\
                        get_identity_label(identity.id)
                    
                    if name == 'Unknown':
                        pass
                    else:                   
                        face_recognition_instance = FaceDB()
                        image = Image.fromarray(cv2.cvtColor(frame,
                                                            cv2.COLOR_BGR2RGB))
                        image_buffer = BytesIO()
                        image.save(image_buffer, format='JPEG')
                        image_data = image_buffer.getvalue()
                        image_file = InMemoryUploadedFile(
                            image_buffer, None, 'img.jpg', 
                            'image/jpeg', len(image_data), None
                        )

                        face_recognition_instance.image = image_file
                        face_recognition_instance.name =  name
                        face_recognition_instance.save()
            
            

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

# Página de streaming
@gzip.gzip_page
def stream(request):

    while True:
        try:
            # Inicializa a câmera
            camera = cv2.VideoCapture(0)  # Use o índice correto se você tiver várias câmeras
            break
        except ValueError:
            print(ValueError)

    return StreamingHttpResponse(generate_frames(camera),
                                 content_type='multipart/x-mixed-replace; boundary=frame')

def search_images(request):
    if request.method == 'POST':
        form = SearchForm(request.POST)
        if form.is_valid():
            search_query = form.cleaned_data['search_query']
            images = FaceDB.objects.filter(name__icontains=search_query)

            #image_names = FaceDB.objects.values_list('name', flat=True)
            #for n in image_names:
            #    print(n)

            #image_paths = FaceDB.objects.values_list('image', flat=True)
            #for path in image_paths:
            #    print(path)

            return render(request, 'search.html',
                           {'images': images, 'form': form})
    else:
        form = SearchForm()
    return render(request, 'search.html', {'form': form})


# Página inicial
def cam(request):
    if request.method == 'POST':
        form = SearchForm(request.POST)
        if form.is_valid():
            search_query = form.cleaned_data['search_query']
            images = FaceDB.objects.filter(name__icontains=search_query)
            print(images)
            return render(request, 'camera_stream.html',
                           {'images': images, 'form': form})
    else:
        print("GERADO")
        form = SearchForm()
    return render(request, 'camera_stream.html', {'form': form})
