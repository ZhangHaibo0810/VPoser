import sys
import os
import json
import torch
import numpy as np
import vposer

def main(args):
    model_load_path = args[1]
    save_dir_path = args[2]

    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    np_save_path = os.path.join(save_dir_path, "vposer_test_data.npz")
    json_save_path = os.path.join(save_dir_path, "vposer_test_data.json")

    print("Load VPoser parameters from: ", os.path.abspath(model_load_path))
    vp = vposer.create(model_load_path, 2)

    tensor_in = torch.rand(1, 32, dtype=torch.float32, requires_grad=True)
    tensor_out = vp.decode(tensor_in)['pose_body']
    tensor_out_norm = tensor_out.norm()
    tensor_out_norm.backward()

    print("Save VPoser data to: ", os.path.abspath(save_dir_path))
    data_np = {
        "VPoserDecoder.in": tensor_in.detach().numpy(),
        "VPoserDecoder.out": tensor_out.detach().numpy(),
        "VPoserDecoder.grad": tensor_in.grad.detach().numpy()
    }
    data_json = {
        "VPoserDecoder.in": tensor_in.detach().numpy().tolist(),
        "VPoserDecoder.out": tensor_out.detach().numpy().tolist(),
        "VPoserDecoder.grad": tensor_in.grad.detach().numpy().tolist()
    }
    
    np.savez(np_save_path, **data_np)
    with open(json_save_path, "w+") as f:
        json.dump(data_json, f, indent=4, sort_keys=True)
    
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemError("Invalid number of arguments!\n"
                          "USAGE: {} <model_load_path> <save_dir_path>".format(sys.argv[0]))
    main(sys.argv)