#Author: Rajnesh Kathuria
import argparse

def get_input_args():
    
    parser = argparse.ArgumentParser(description='Enter image path and checkpoint file. Optionally enter top_k, category mapping file, gpu')
    parser.add_argument("image_path", help="Required argument: Image path and name")
    parser.add_argument("checkpoint", help="Required argument: Checkpoint file")
    parser.add_argument("--top_k", default=1, help="top K (default=1)",  type=int)
    parser.add_argument("--category_names", default="cat_to_name.json", help="category mapping file(default: cat_to_name.json)")
    parser.add_argument("-gpu", help="will run on GPU if this argument is set", action="store_true")
    
    args = parser.parse_args()

    return args
