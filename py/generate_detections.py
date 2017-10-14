# vim: expandtab:ts=4:sw=4
import numpy as np
import cv2

#import gen.enc as enc
import enc as enc
from io import StringIO

class Gd(object):
    def __init__(self):
        print("hahahah0")
        self.image_shape = 128, 64, 3
        print("hahahah1")
        self.enc = enc.Enc()
        
    def _extract_image_patch(self, image, bbox, patch_shape):
        """Extract image patch from bounding box.
    
        Parameters
        ----------
        image : ndarray
            The full image.
        bbox : array_like
            The bounding box in format (x, y, width, height).
        patch_shape : Optional[array_like]
            This parameter can be used to enforce a desired patch shape
            (height, width). First, the `bbox` is adapted to the aspect ratio
            of the patch shape, then it is clipped at the image boundaries.
            If None, the shape is computed from :arg:`bbox`.
    
        Returns
        -------
        ndarray | NoneType
            An image patch showing the :arg:`bbox`, optionally reshaped to
            :arg:`patch_shape`.
            Returns None if the bounding box is empty or fully outside of the image
            boundaries.
    
        """
        bbox = np.array(bbox)
        if patch_shape is not None:
            # correct aspect ratio to patch shape
            target_aspect = float(patch_shape[1]) / patch_shape[0]
            new_width = target_aspect * bbox[3]
            bbox[0] -= (new_width - bbox[2]) / 2
            bbox[2] = new_width
    
        # convert to top left, bottom right
        bbox[2:] += bbox[:2]
        bbox = bbox.astype(np.int)
    
        # clip at image boundaries
        bbox[:2] = np.maximum(0, bbox[:2])
        bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
        if np.any(bbox[:2] >= bbox[2:]):
            return None
        sx, sy, ex, ey = bbox
        image = image[sy:ey, sx:ex]
        image = cv2.resize(image, patch_shape[::-1])
    
        return image
    
    
    def preEncode(self, model_filename):
        print("hahahah2")
        self.enc.preEncode(self.image_shape, 128, model_filename, "cosine")
        print("hahahah")
        return 1
        #def encoder(image, box):
        #    image_patches = []
        #    t1 = int(round(time.time() * 1000))
        #    patch = extract_image_patch(image, box, image_shape[:2])
        #    t2 = int(round(time.time() * 1000))
        #    print("extract feature cost time:%s" % str(t2-t1))
        #    if patch is None:
        #        print("WARNING: Failed to extract image patch: %s." % str(box))
        #        patch = np.random.uniform(
        #            0., 255., image_shape).astype(np.uint8)
        #    image_patches.append(patch)
        #    image_patches = np.asarray(image_patches)
        #    return image_encoder(image_patches)
    
    def encode(self, image, boxes):
        image_patches = []
        for box in boxes:
            patch = self._extract_image_patch(image, box, self.image_shape[:2])
            if patch is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., self.image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        #print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
        return self.enc.encode(image_patches)
    
    def encodeForCpp(self, img, cols, rows, boxesStr):
        #print("mmmmmmm0")
        img_data = np.reshape(img, (cols, rows, 3)).astype(np.uint8)
        #cv2.imshow("img_data", img_data)
        #cv2.waitKey()
        #print("mmmmmmm1")
        #print(boxesStr)
        #a = StringIO(unicode(boxesStr))
        #print(a)
        #print("mmmmmmm1.1")
        boxes = np.loadtxt(StringIO(unicode(boxesStr)))
        #print(boxes.shape)
        if(boxes.shape == (4,)):
            boxes = boxes[np.newaxis, :]
        #print(boxes.shape)    
        #print("mmmmmmm2")
        return self.encode(img_data, boxes)
        #tmp = self.encode(img_data, boxes)
        #print("mmmmmmm3")
        #print(tmp)
        #return tmp
        
        
        
        
        
        
        
