import numpy as np
import time
import cv2
import os
import imutils
import subprocess
from gtts import gTTS 
from pydub import AudioSegment
AudioSegment.converter = "C:/Users/jasonyip184/Desktop/yolo-object-detection/ffmpeg-20181202-72b047a-win64-static/bin/ffmpeg.exe"

# load the COCO class labels our YOLO model was trained on
LABELS = open("yolo-coco/coco.names").read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("yolo-coco/yolov3.cfg", "yolo-coco/yolov3.weights")
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture("in.mp4")
first = True
detected = []
# initialize our video writer
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter("video.avi", fourcc, 30, (1280, 720), True)
frame_count = 0

# loop over frames from the video file stream
while True:
	print(frame_count)
	frame_count += 1
	# read the next frame from the file
	(grabbed, frame) = vs.read()
	frame = cv2.flip(frame,1)
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	if frame_count % 2 == 0:
		(H, W) = frame.shape[:2]

		# construct a blob from the input frame and then perform a forward
		# pass of the YOLO object detector, giving us our bounding boxes
		# and associated probabilities
		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
			swapRB=True, crop=False)
		net.setInput(blob)
		start = time.time()
		layerOutputs = net.forward(ln)

		# initialize our lists of detected bounding boxes, confidences,
		# and class IDs, respectively
		boxes = []
		confidences = []
		classIDs = []
		centers = []

		# loop over each of the layer outputs
		for output in layerOutputs:
			# loop over each of the detections
			for detection in output:
				# extract the class ID and confidence (i.e., probability)
				# of the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]

				# filter out weak predictions by ensuring the detected
				# probability is greater than the minimum probability
				if confidence > 0.5:
					# scale the bounding box coordinates back relative to
					# the size of the image, keeping in mind that YOLO
					# actually returns the center (x, y)-coordinates of
					# the bounding box followed by the boxes' width and
					# height
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")

					# use the center (x, y)-coordinates to derive the top
					# and and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))

					# update our list of bounding box coordinates,
					# confidences, and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)
					centers.append((centerX, centerY))

		# apply non-maxima suppression to suppress weak, overlapping
		# bounding boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

		texts = []

		# ensure at least one detection exists
		if len(idxs) > 0:
			# loop over the indexes we are keeping
			for i in idxs.flatten():
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])

				# draw a bounding box rectangle and label on the frame
				color = [int(c) for c in COLORS[classIDs[i]]]
				cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
				text = "{}: {:.4f}".format(LABELS[classIDs[i]],
					confidences[i])
				cv2.putText(frame, text, (x, y - 5),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

				if frame_count % 30 == 0:
					label = LABELS[classIDs[i]]
					if label not in detected:
						# find positions
						centerX, centerY = centers[i][0], centers[i][1]
						
						if centerX <= W/3:
							W_pos = "left "
						elif centerX <= (W/3 * 2):
							W_pos = "center "
						else:
							W_pos = "right "
						
						if centerY <= H/3:
							H_pos = "top "
						elif centerY <= (H/3 * 2):
							H_pos = "mid "
						else:
							H_pos = "bottom "
						texts.append(H_pos + W_pos + label)
						detected.append(label)
		
		if frame_count % 30 == 0:
			if len(texts) > 0:
				silence = AudioSegment.silent(duration=0.01*1000)
				description = ', '.join(texts)
				tts = gTTS(description, lang='en')
				tts.save('tts.mp3')
				tts = AudioSegment.from_mp3("tts.mp3")
				if first:
					audio = tts
				else:
					audio = AudioSegment.from_mp3("audio.mp3")
					audio = audio + silence + tts
			else:
				silence = AudioSegment.silent(duration=1*1000)
				if first:
					audio = silence
				else:
					audio = AudioSegment.from_mp3("audio.mp3")
					audio = audio + silence

			audio.export("audio.mp3", format="mp3")
			first = False

	# write the output frame to disk
	writer.write(frame)

# release the file pointers
writer.release()
vs.release()
cv2.destroyAllWindows()
os.remove("tts.mp3")

cmd = 'ffmpeg -i video.avi -i audio.mp3 -c copy output.mp4'
subprocess.call(cmd, shell=True)
