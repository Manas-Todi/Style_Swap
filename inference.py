from inference_loop import main
import os


main(10, 0.05, clip_w=1, img_w=30,reg_w=1,body_shape_w=10,head_w=1,stitch=True, image_path='test_images/test2.jpg',text_prompt="Grained leather crop top in vivid fuchsia pink.")
