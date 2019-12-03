'''
Face Segmentation labeling tool.

Generate Face mask by mouse painting.
Green Brush for adding new Face region.
Red Brush for removinng region.

Keys:
  Crop mode: controled by 'crop_flag', default is True; For Large input image size, crop Face ROI firstly
             Caculating the IN-Polygon Pixels is time consuming for large image size.
    ESC  - exit, crop selected Rectangular region and enter into Segmentation mode

  Segmentation mode:
    a     - switch to Green Brush, adding new Face region
    r     - switch to Red Brush, removing region
    ESC   - exit and save

Mouse:
  Left button down and hold: Drawing(Segmentation mode) or Create Rectangular box(Crop mode)
  Left button up(Release): Finish a bounding box selection(Crop mode, you can re-draw a box just by press and hold Left button again),
                           Finish selecting a region, calculating the selected pixel area. 
'''
import numpy as np 
import cv2
from pathlib import Path
from matplotlib.path import Path as pltPath
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import shutil

class MaskSeg:
    def __init__(self, img_path, mask_save_path, masked_frames_path, old_mask_file, crop_flag = True):
        self.ori_img = cv2.imread(str(img_path.absolute())).astype(np.float32)
        self.img_path = img_path
        self.mask_save_path = mask_save_path
        self.masked_frames_path = masked_frames_path
        self.crop_flag = crop_flag
        if self.ori_img is None:
            print(f'Failed to load image file: {img_path.absolute()}')
            sys.exit(1)
        if old_mask_file != None:
            self.old_mask = cv2.imread(str(old_mask_file.absolute())).astype(np.float32)
        else:
            self.old_mask = None

        self.img = self.ori_img.copy()
        self.cropped_old_mask = None
        if self.crop_flag:
            self.crop_resized_img, self.crop_resize_ratio = self.resize_ratio(self.img)
            self.crop_rec_lt = None
            self.crop_rec_rb = None
            self.crop_trans_mat = np.eye(3)[0:2, :]
            self.crop_show_img = self.crop_resized_img.copy()
            self.crop_windowname = str(self.img_path.name).strip().split('.')[0] + '_crop'
            self.prev_pt = None
            self.img, self.cropped_old_mask = self.crop_image()
            cv2.destroyAllWindows()

        self.windowname = str(img_path.name).strip().split('.')[0]
        cv2.namedWindow(self.windowname, cv2.WINDOW_NORMAL)
        if self.cropped_old_mask is not None:
            self.mask = self.cropped_old_mask
        else:
            self.mask = np.zeros_like(self.img)
    
        self.add_flag = True
        self.prev_pt = None
        self.verts = []

        self.overlay_img = self.overlay()
        self.show(self.overlay_img)
        cv2.setMouseCallback(self.windowname, self.on_mouse)
        self.on_keyboard()
        print('Done')
        cv2.destroyAllWindows()

    def show(self, show_img):
        cv2.imshow(self.windowname, show_img.astype(np.uint8))

    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
        elif event == cv2.EVENT_LBUTTONUP:
            #add or remove region and reset the flag
            self.prev_pt = None
            if len(self.verts) > 2:
                new_region = self.get_mask_region()
                if self.add_flag:
                    self.mask += new_region
                    self.mask = np.clip(self.mask, 0., 255.)
                else:
                    self.mask -= new_region
                    self.mask = np.clip(self.mask, 0., 255.)
            self.overlay_img = self.overlay()

            self.verts = []
            self.show(self.overlay_img)

        if self.prev_pt and flags & cv2.EVENT_FLAG_LBUTTON:
            if self.add_flag:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.line(self.overlay_img, self.prev_pt, pt, color, 1)
            self.prev_pt = pt
            self.show(self.overlay_img)
            self.verts.append(pt)
    
    def on_keyboard(self):
        while True:
            ch = cv2.waitKey()
            if ch == 27:
                #save mask and exit
                mask_file_name = self.mask_save_path.joinpath('mask_' + self.img_path.name)
                temp_masked_img = self.mask/255. * self.img
                cv2.imwrite(str(self.masked_frames_path.joinpath('temp_masked')) + self.img_path.name, temp_masked_img)
                if self.crop_flag:
                    self.mask = cv2.warpAffine(self.mask.astype(np.uint8), self.crop_trans_mat, (self.ori_img.shape[1], self.ori_img.shape[0]))
                if self.old_mask is not None:
                    self.mask = np.clip(0, 255, self.mask + self.old_mask)
                cv2.imwrite(str(mask_file_name.absolute()), self.mask)
                masked_img = self.mask/255. * self.ori_img
                file_name = self.masked_frames_path.joinpath('masked_' + self.img_path.name)
                cv2.imwrite(str(file_name), masked_img)
                break

            if ch == ord('a'):
                #add a new region to mask
                self.add_flag = True

            if ch == ord('r'):
                #remove an region from mask
                self.add_flag = False

    def get_mask_region(self):
        verts = self.verts
        verts.append(verts[0])
        verts = [v[::-1] for v in verts]
        polygon = pltPath(verts, closed = True)

        x, y =np.meshgrid(np.arange(self.img.shape[0]), np.arange(self.img.shape[1]))
        points = list(zip(x.flatten(), y.flatten()))

        #start_time = time.time()
        grid = polygon.contains_points(points)
        new_mask = np.ones((self.img.shape[0], self.img.shape[1], 3))*255
        for i, c in enumerate(points):
            new_mask[c[0], c[1], :] *= grid[i]
        #print(f"Time is {time.time() - start_time}")

        return new_mask

    def overlay(self, alpha = 0.3):
        red_mask = self.mask.copy()
        red_mask[:, :, 0] *= 0
        red_mask[:, :, -1] *= 0
        overlay_img = red_mask*alpha + self.img*(self.mask/255.)*(1-alpha) + self.img*(1- self.mask/255.)
  
        return overlay_img
    
    def crop_image(self):
        cv2.namedWindow(self.crop_windowname, cv2.WINDOW_NORMAL)
        cv2.imshow(self.crop_windowname, self.crop_show_img.astype(np.uint8))

        cv2.setMouseCallback(self.crop_windowname, self.crop_on_mouse)

        while True:
            ch = cv2.waitKey()
            if ch == 27 and self.crop_rec_lt != None and self.crop_rec_rb != None:
                (t, l), (b, r) = self.correct_lt_rb()
                cropped_img = self.img[l:r, t:b]
                if self.old_mask is not None:
                    old_cropped_mask = self.old_mask[l:r, t:b]
                else:
                    old_cropped_mask = None
                self.crop_trans_mat[0,2] = t
                self.crop_trans_mat[1,2] = l
                break
        
        return cropped_img, old_cropped_mask

    def crop_on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.prev_pt = pt
            if self.crop_rec_rb != None:
                self.crop_rec_lt = None
                self.crop_rec_rb = None
            if self.crop_rec_lt == None:
                self.crop_rec_lt = pt
        elif event == cv2.EVENT_LBUTTONUP:
            self.crop_rec_rb = self.prev_pt
            self.prev_pt = None

        if self.prev_pt and flags & cv2.EVENT_FLAG_LBUTTON:
            cv2.rectangle(self.crop_show_img, self.crop_rec_lt, self.prev_pt, (0,255,0), 2)
            self.prev_pt = pt
            cv2.imshow(self.crop_windowname, self.crop_show_img.astype(np.uint8))
            self.crop_show_img = self.crop_resized_img.copy()
    
    def correct_lt_rb(self):
        t, l = int(self.crop_rec_lt[0]/self.crop_resize_ratio), int(self.crop_rec_lt[1]/self.crop_resize_ratio)
        b, r = int(self.crop_rec_rb[0]/self.crop_resize_ratio), int(self.crop_rec_rb[1]/self.crop_resize_ratio)

        re_t, re_b, re_l, re_r = 0, 0, 0, 0
        if b >= t:
            re_t, re_b = t, b
        else:
            re_t, re_b = b, t
        
        if r >= l:
            re_l, re_r = l, r
        else:
            re_l, re_r = r, l
        
        return (re_t, re_l), (re_b, re_r)
    
    def resize_ratio(self, img):
        img_H, img_W = img.shape[0], img.shape[1]
        if img_H > img_W:
            if img_H > 720:
                resize_ratio = 720/img_H
                target_W = int(resize_ratio * img_W)
                return cv2.resize(img, (target_W, 720)), resize_ratio
            else:
                return img, 1.0
        else:
            if img_W > 1280:
                resize_ratio = 1280/img_W
                target_H = int(resize_ratio * img_H)
                return cv2.resize(img, (1280, target_H)), resize_ratio
            else:
                return img, 1.0
        

def seg_main(frames_path, mask_save_path, masked_frame_path, old_mask_path = None):
    images_list = []
    images_list.extend(frames_path.glob('*.jpg'))
    images_list.extend(frames_path.glob('*.png'))
    for f in images_list:
        old_mask_file = None
        if old_mask_path != None:
            old_mask_file = old_mask_path.joinpath('mask_' + f.name)
            if not old_mask_file.is_file():
                print(f'Old mask {old_mask_file.absolute()} does not exsit!!')
                print(f'Init mask with empty.')
                old_mask_file = None
        MaskSeg(f.absolute(), mask_save_path, masked_frame_path, old_mask_file)
        shutil.move(f.absolute().as_posix(), masked_frame_path.absolute().as_posix())


if __name__ == '__main__':
    print(__doc__)
    frames_path = Path(__file__).parent.joinpath('V1_frames')  #input face images

    mask_save_path = Path(__file__).parent.joinpath('V1_mask') #saved mask dir
    if not mask_save_path.exists():
        mask_save_path.mkdir(parents=True, exist_ok=True)

    masked_frame_path = Path(__file__).parent.joinpath('V1_masked_frames') #saved segmented faces dir
    if not masked_frame_path.exists():
        masked_frame_path.mkdir(parents=True, exist_ok=True)

    #old_mask_path = Path(__file__).parent.joinpath('V1_old_mask') #Old mask for loading
    old_mask_path = None 
    seg_main(frames_path, mask_save_path, masked_frame_path, old_mask_path)