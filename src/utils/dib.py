import cv2

def dib_cajas(frame, results, class_names=None, color=(0,255,0)):
    for r in results:
        for b in r.boxes:
            x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
            cls = int(b.cls[0])
            id_ = int(b.id[0]) if b.id is not None else -1
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            label = f"{class_names[cls] if class_names else cls}"
            if id_ >= 0: label += f" #{id_}"
            cv2.putText(frame,label,(x1,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
    return frame

def dib_lineas(frame, lines, color=(255,0,0)):
    for (p1,p2) in lines:
        cv2.line(frame,p1,p2,color,2)
    return frame

def dib_panel_cont(frame, counts: dict, origin=(15,25)):
    x,y = origin
    cv2.putText(frame,"Conteo:",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    y += 22
    for k,v in counts.items():
        cv2.putText(frame,f"{k}: {v}",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
        y += 20
    return frame
