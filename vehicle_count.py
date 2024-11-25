import cv2
from collections import defaultdict
import supervision as sv
from ultralytics import YOLO
import datetime
import numpy as np


model = YOLO('yolov8m.pt')

class_mapping = {
    2: 'Araba', 3: 'Motorsiklet', 5: 'Otobüs', 7: 'Kamyon/Tir',
}


cap = cv2.VideoCapture("C:\\Users\Muhammed Fatih\\Documents\\Vehicle_counter\\traffic.mp4")


START = sv.Point(220, 500)
END = sv.Point(1040, 500)



track_history = defaultdict(lambda: [])


crossed_objects = {}
car_objects = {}
motorcycle_objects = {}
bus_objects = {}
truck_objects = {}


video_info = sv.VideoInfo.from_video_path("C:\\Users\Muhammed Fatih\\Documents\\Vehicle_counter\\traffic.mp4")
with sv.VideoSink("C:\\Users\Muhammed Fatih\\Documents\\Vehicle_counter\\traffic_Test.mp4", video_info) as sink:

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.track(frame, classes=[2, 3, 5, 7], persist=True, save=True, tracker="bytetrack.yaml")


            x = datetime.datetime.now()
            year = x.strftime("%Y")
            month = x.strftime("%B")
            day = x.strftime("%A")
            hour = x.strftime("%X")

            # workbook = xlsxwriter.Workbook( "benim_dosyam.xlsx" )
            # worksheet = workbook.add_worksheet()

            # worksheet.write(0,0,"Year")
            # worksheet.write(0,1,"Month")
            # worksheet.write(0,2,"Day")
            # worksheet.write(0,3,"Hour")
            # worksheet.write(0,4,"Car")
            # worksheet.write(0,5,"Motorcycle")
            # worksheet.write(0,6,"Bus")
            # worksheet.write(0,7,"Truck")
            # worksheet.write(0,8,"Totally")

            if  results[0].boxes.id != None:
        
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                clss = results[0].boxes.cls.cpu().tolist()
                names = results[0].names

                for result in results:
                    for cls_id, custom_label in class_mapping.items():
                        if cls_id in result.names: 
                            result.names[cls_id] = custom_label 


                
                annotated_frame = results[0].plot()   
                annotated_frame_array = np.array(annotated_frame)
                annotated_frame_writable = annotated_frame_array.copy()
                detections = sv.Detections.from_ultralytics(results[0])

             
                for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  
                    if len(track) > 30:  
                        track.pop(0)

                    
                    if START.x < x < END.x and abs(y - START.y) < 10: 
                        if track_id not in crossed_objects:
                            crossed_objects[track_id] = True
                            if class_id == 2:
                                car_objects[track_id] = True
                            elif class_id == 3:
                                motorcycle_objects[track_id] = True
                            elif class_id == 5:
                                bus_objects[track_id] = True
                            elif class_id == 7:
                                truck_objects[track_id] = True



                        
                        cv2.rectangle(annotated_frame_writable, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)
            else:
               annotated_frame = results[0].plot()
               annotated_frame_array = np.array(annotated_frame)
               annotated_frame_writable = annotated_frame_array.copy()
               



           
            cv2.line(annotated_frame_writable, (START.x, START.y), (END.x, END.y), (0, 255, 0), 2)

           
            count_car = f"Araba: {len(car_objects)}"
            cv2.putText(annotated_frame_writable, count_car, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            count_motorcycle = f"Motorsiklet: {len(motorcycle_objects)}"
            cv2.putText(annotated_frame_writable, count_motorcycle, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            count_bus = f"Otobus: {len(bus_objects)}"
            cv2.putText(annotated_frame_writable, count_bus, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            count_truck = f"Kamyon-Tir: {len(truck_objects)}"
            cv2.putText(annotated_frame_writable, count_truck, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            count_text = f"Toplam: {len(crossed_objects)}"
            cv2.putText(annotated_frame_writable, count_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


            #worksheet.write(1,0,year)
            #worksheet.write(1,1,month)
            #worksheet.write(1,2,day)
            #worksheet.write(1,3,hour)
            #worksheet.write(1,4,len(car_objects))
            #worksheet.write(1,5, len(motorcycle_objects))
            #worksheet.write(1,6,len(bus_objects))
            #worksheet.write(1,7,len(truck_objects))
            #worksheet.write(1,8,len(crossed_objects)

           
            print(f"Yıl: {year}")
            print(f"Ay: {month}")
            print(f"Gün: {day}")
            print(f"Saat: {hour}")
            print(f"Araba: {len(car_objects)}")
            print(f"Motorsiklet: {len(motorcycle_objects)}")
            print(f"Otobüs: {len(bus_objects)}")
            print(f"Komyaon-Tır: {len(truck_objects)}")
            print(f"Toplam Araç Sayısı: {len(crossed_objects)}")

            
            sink.write_frame(annotated_frame_writable)


        else:
            break

file1 = open("MyFile1.txt","a")
print(f"Yil: {year}  Ay: {month}  Gun: {day}  Saat: {hour}  Araba: {len(car_objects)}  Motorsiklet: {len(motorcycle_objects)}  Otobus: {len(bus_objects)} Komyon-Tir: {len(truck_objects)} Toplam Arac Sayisi: {len(crossed_objects)}", file=file1)
file1.close()
# workbook.close()
# Release the video capture
cap.release()