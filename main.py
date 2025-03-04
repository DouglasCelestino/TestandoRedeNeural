import cv2
import os
import matplotlib
matplotlib.use('Agg')  # Usa backend sem interface gráfica
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Caminhos
modelo_path = "D:/8Semestre/PFE/TestandoRedeNeural/runs/detect/train/weights/best.pt"
imagens_dir = "assets/images/teste"
output_dir = "output"

# Garante que a pasta de saída existe
os.makedirs(output_dir, exist_ok=True)

def processar_imagens():
    """Carrega o modelo e faz a inferência nas imagens do diretório, gerando relatório"""
    # Carregar o modelo YOLOv8
    modelo = YOLO(modelo_path)

    # Lista todas as imagens na pasta
    imagens = sorted([os.path.join(imagens_dir, img) for img in os.listdir(imagens_dir) if img.endswith((".jpg", ".png", ".jpeg"))])

    # Loop para inferência e exibição
    for img_path in imagens:
        # Carrega a imagem
        imagem = cv2.imread(img_path)
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

        # Faz a predição
        resultados = modelo(img_path)

        # Nome base da imagem para salvar arquivos
        img_name = os.path.basename(img_path)
        report_path = os.path.join(output_dir, f"relatorio_{img_name}.txt")

        # Criar e escrever no arquivo de relatório
        with open(report_path, "w", encoding="utf-8") as report_file:
            report_file.write(f"Relatório de Predição - {img_name}\n")
            report_file.write("=" * 50 + "\n")

            # Desenha as detecções na imagem
            for result in resultados:
                boxes = result.boxes.xyxy  # Coordenadas das bounding boxes
                confs = result.boxes.conf  # Confiança
                classes = result.boxes.cls  # Classes detectadas

                for box, conf, cls in zip(boxes, confs, classes):
                    x1, y1, x2, y2 = map(int, box)  # Coordenadas da bounding box
                    class_name = modelo.names[int(cls)]  # Nome da classe
                    conf_percent = conf * 100  # Confiança em porcentagem
                    label = f"{class_name} {conf_percent:.2f}%"

                    # Adiciona ao relatório
                    report_file.write(f"Classe: {class_name}\n")
                    report_file.write(f"Confiança: {conf_percent:.2f}%\n")
                    report_file.write(f"Bounding Box: ({x1}, {y1}) -> ({x2}, {y2})\n")
                    report_file.write("-" * 50 + "\n")

                    # Desenha bounding box e rótulo na imagem
                    cv2.rectangle(imagem, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(imagem, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            print(f"Relatório gerado: {report_path}")

        # Criar figura do Matplotlib e salvar a imagem
        plt.figure(figsize=(8, 6))
        plt.imshow(imagem)
        plt.axis("off")
        plt.title(f"Predição: {img_name}")

        # Define caminho de saída e salva a imagem
        output_img_path = os.path.join(output_dir, f"output_{img_name}")
        plt.savefig(output_img_path, bbox_inches='tight')
        plt.close()  # Fecha a figura para liberar memória
        
        print(f"Imagem salva em: {output_img_path}")

# Executa apenas se o script for rodado diretamente
if __name__ == "__main__":
    processar_imagens()
