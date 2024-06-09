import cv2
import numpy as np
import argparse
from numpy_tps_transform import transformer, draw_mesh_on_warp


def get_rigid_mesh(height, width, grid_w, grid_h): 
    ww = np.matmul(np.ones([grid_h+1, 1]), np.expand_dims(np.linspace(0., float(width), grid_w+1), 0))
    hh = np.matmul(np.expand_dims(np.linspace(0.0, float(height), grid_h+1), 1), np.ones([1, grid_w+1]))
    
    ori_pt = np.concatenate((np.expand_dims(ww, 2), np.expand_dims(hh,2)), axis=2)
    return ori_pt[np.newaxis, :]  ###batchsize=1

def get_norm_mesh(mesh, height, width):
    mesh_w = mesh[...,0]*2./float(width) - 1.
    mesh_h = mesh[...,1]*2./float(height) - 1.
    norm_mesh = np.stack([mesh_w, mesh_h], axis=3) 
    
    return norm_mesh.reshape((1, -1, 2)) 

class RecRecNet():
    def __init__(self, modelpath):
        self.net = cv2.dnn.readNet(modelpath)
        self.grid_w, self.grid_h = 8, 8
        self.input_width, self.input_height = 256, 256
        self.output_names = self.net.getUnconnectedOutLayersNames()
        self.rigid_mesh = get_rigid_mesh(self.input_height, self.input_width, self.grid_w, self.grid_h)

    def detect(self, srcimg):
        img = cv2.resize(srcimg, dsize=(self.input_width, self.input_height))
        img = img.astype(np.float32) / 127.5 - 1.0
        blob = cv2.dnn.blobFromImage(img)
        self.net.setInput(blob)
        offset = self.net.forward(self.output_names)[0]
        
        mesh_motion = offset.reshape(self.grid_h+1, self.grid_w+1, 2)
        ori_mesh = self.rigid_mesh + mesh_motion

        norm_rigid_mesh = get_norm_mesh(self.rigid_mesh, self.input_height, self.input_width)
        norm_ori_mesh = get_norm_mesh(ori_mesh, self.input_height, self.input_width)
        
        output_tps = transformer(blob, norm_rigid_mesh, norm_ori_mesh, (self.input_height, self.input_width))

        rectangling_np = ((output_tps[0]+1)*127.5).transpose(1,2,0)
        input_np = ((blob[0]+1)*127.5).transpose(1,2,0)
        ori_mesh_np = ori_mesh[0]

        return rectangling_np.astype(np.uint8), input_np, ori_mesh_np
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", type=str, default='testimgs/10.jpg')
    args = parser.parse_args()

    mynet = RecRecNet('model_deploy.onnx')
    srcimg = cv2.imread(args.imgpath)

    rectangling_np, input_np, ori_mesh_np = mynet.detect(srcimg)

        
    # path = "rect.jpg"
    # cv2.imwrite(path, rectangling_np)
    
    input_with_mesh = draw_mesh_on_warp(input_np, ori_mesh_np, mynet.grid_w, mynet.grid_h)
    # path = "mesh.jpg"
    # cv2.imwrite(path, input_with_mesh)

    cv2.namedWindow('srcimg', 0)
    cv2.imshow('srcimg', srcimg)
    cv2.namedWindow('rect', 0)
    cv2.imshow('rect', rectangling_np)
    cv2.namedWindow('mesh', 0)
    cv2.imshow('mesh', input_with_mesh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()