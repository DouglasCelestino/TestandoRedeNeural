from ultralytics import YOLO

# =============================================================================
# Modelos disponíveis para YOLOv8 - Detecção de Objetos
# -----------------------------------------------------------------------------
# Modelo        | Tamanho       | Parâmetros  | Requisitos de VRAM | Uso Ideal
# -----------------------------------------------------------------------------
# yolov8n.pt    | Nano         | 3.2M        | ~2GB VRAM          | Rápido, ideal para dispositivos fracos
# yolov8s.pt    | Small        | 11.2M       | ~3GB VRAM          | Bom equilíbrio entre velocidade e precisão
# yolov8m.pt    | Medium       | 25.9M       | ~4GB VRAM          | Melhor precisão, mas mais pesado
# yolov8l.pt    | Large        | 43.7M       | ~6GB VRAM          | Alta precisão, exige mais memória
# yolov8x.pt    | Extra-Large  | 68.2M       | ~8GB+ VRAM         | Máxima precisão, precisa de GPU forte
# =============================================================================


from ultralytics import YOLO

def train_model():
    # Carregar modelo YOLOv8 pré-treinado
    model = YOLO("yolov8n.pt")

    model.train(
        data="D:/8Semestre/PFE/TestandoRedeNeural/dataset/PPE_cdp.v1-model.yolov8/data.yaml",
        epochs=5,  # Aumentando para aprendizado melhor
        imgsz=1024,
        batch=4,  # Reduzindo batch para evitar estouro de VRAM
        workers=2,  # Menos workers para evitar sobrecarga
        device="cuda",  # Agora forçando uso da GPU 
        verbose=True
    )

if __name__ == "__main__":
    try:
        train_model()
        print("Treinamento finalizado! Confira os resultados")
    except Exception as e:
        print(e)