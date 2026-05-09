import gradio as gr
from AI_Detect import process_image

with gr.Blocks(title="🧠 Brain Tumor Detector") as demo:
    gr.Markdown(
        """
        # Детектор опухолей мозга по МРТ-снимкам
        Загрузите изображение для автоматической локализации патологии. 
        Результат можно сохранить через правый клик или меню справа от картинки.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="📤 Загрузите МРТ-снимок",
                type="numpy",
                height=400
            )
            threshold = gr.Slider(
                minimum=0.1, maximum=0.95, value=0.5, step=0.05,
                label=" Порог уверенности"

            )
            detect_btn = gr.Button("🔍 Обработать снимок", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="📥 Результат",
                type="numpy",
                height=400
                # show_download_button удалён (устарел в Gradio 4+)
            )
            report_box = gr.Textbox(
                label="📊 Отчёт детекции",
                lines=8,
                interactive=False
                # show_copy_button удалён (встроен автоматически в Gradio 4+)
            )

    detect_btn.click(
        fn=process_image,
        inputs=[input_image, threshold],
        outputs=[output_image, report_box]
    )

    input_image.change(
        fn=process_image,
        inputs=[input_image, threshold],
        outputs=[output_image, report_box]
    )

    gr.Markdown(
        """
        ---
        ** Модель:** Faster R-CNN + ResNet-50 FPN | **Параметры:** ~41.8M 
        """
    )

if __name__ == '__main__':
    print("\n🌐 Запуск сервера...")
    print("📍 Локально: http://localhost:7860")
    print("📍 В сети: http://<ВАШ_IP>:7860")
    print("⏹️ Остановка: Ctrl+C\n")

    custom_css = """
    .footer {visibility: hidden;}
    .gradio-container {max-width: 900px !important; margin: 0 auto;}
    """
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        css=custom_css
    )