#!/bin/sh
python main_task_2.py --image "sample_input_images/chair.jpg" --class "chair" --output "task_2_inter_output_images/chair.jpg"
python main_task_1.py --image "sample_input_images/chair(1).jpg" --class "chair" --output "task_2_inter_output_images/chair(1).jpg"
python main_task_1.py --image "sample_input_images/flower vase.jpg" --class "flower vase" --output "task_2_inter_output_images/flower vase.jpg"
python main_task_1.py --image "sample_input_images/lamp.jpg" --class "lamp" --output "task_2_inter_output_images/lamp.jpg"
python main_task_1.py --image "sample_input_images/laptop.jpg" --class "laptop" --output "task_2_inter_output_images/laptop.jpg"
python main_task_1.py --image "sample_input_images/office chair.jpg" --class "office chair" --output "task_2_inter_output_images/office chair.jpg"
python main_task_1.py --image "sample_input_images/sofa.jpg" --class "sofa" --output "task_2_inter_output_images/sofa.jpg"
python main_task_1.py --image "sample_input_images/table.jpg" --class "table" --output "task_2_inter_output_images/table.jpg"

