import cv2
import numpy as np
import math 
import numba
__all__ =[
   "SelectShape",
   "find_formula_line",
   "OpeningRectangle",
   "ClosingRectangle",
   "FillUp",
   "SelectShapeStd",
   "ConvexHull",
   "ErodeRectangle",
   "line_intersection",
   "SelectContour",
   "drawContourMask"

]
__doc__ = [
   
    
]

class Region ():
    """
        "This is class about region process such as"
    
    """
    @staticmethod
    def SelectShapeStd(Region, mode = "area"):
        numLabels, labels, stats, _ = cv2.connectedComponentsWithStats(Region, 8, cv2.CV_32S)
        h, w = Region.shape

        SelectedRegion = np.zeros((h, w), np.uint8)

        if numLabels <= 1:
            return SelectedRegion, 0
        if mode == "area":
            features = list(stats[1:,4])
            
        elif mode == "width":
            features = list(stats[1:,2])   
            
        elif mode == "height":
            features = list(stats[1:,3])   
            
        max_value  = max(features)
        index = features.index(max_value) + 1
        print(f'index:{index}')
        ObjectSelected = np.array(labels, dtype=np.uint8)

        ObjectSelected[index == labels] = 255
        ObjectSelected[index != labels] = 0

        return ObjectSelected
        
    @staticmethod
    def OpeningRectangle(image, width, height):
        mask = cv2.getStructuringElement(cv2.MORPH_RECT, ksize = (width, height))
        erosion = cv2.erode(image, mask, iterations = 1, borderType = cv2.BORDER_DEFAULT, borderValue = 0)
        opening = cv2.dilate(erosion, mask, iterations = 1, borderType = cv2.BORDER_DEFAULT, borderValue = 0)
        return opening

    @staticmethod
    def ClosingRectangle(image, width, height):
        mask = cv2.getStructuringElement(cv2.MORPH_RECT, ksize = (width, height))
        dilation = cv2.dilate(image, mask, iterations = 1, borderType = cv2.BORDER_DEFAULT, borderValue = 0)
        closing = cv2.erode(dilation, mask, iterations = 1, borderType = cv2.BORDER_DEFAULT, borderValue = 0)
        return closing
    
    @staticmethod
    def OpeningCircle(image, radius):
        mask = cv2.getStructuringElement(cv2.MORPH_CROSS, ksize = (2*radius + 1, 2*radius + 1))
        erosion = cv2.erode(image, mask , iterations = 1, borderType = cv2.BORDER_DEFAULT, borderValue = 0)
        opening = cv2.dilate(erosion, mask , iterations = 1, borderType = cv2.BORDER_DEFAULT, borderValue = 0)
        return opening
    
    @staticmethod
    def ClosingCircle(image, radius):
        mask = cv2.getStructuringElement(cv2.MORPH_CROSS, ksize = (2*radius + 1, 2*radius +1))
        dilation = cv2.dilate(image, mask , iterations = 1, borderType = cv2.BORDER_DEFAULT, borderValue = 0)
        closing = cv2.erode(dilation, mask , iterations = 1, borderType = cv2.BORDER_DEFAULT, borderValue = 0)
        return closing

    @staticmethod
    def FillUp(region):
        region =region.astype(np.uint8)
        # add padding
        im_floodfill = cv2.copyMakeBorder(region, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)

        # Create mask (ROI)
        h, w = im_floodfill.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

        cv2.floodFill(im_floodfill, mask, (0,0), 255, flags = 4)
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        im_floodfill_inv = im_floodfill_inv[1:(1 + region.shape[0]), 1:(1 + region.shape[1])]

        RegionFillUp = region | im_floodfill_inv
        return RegionFillUp

    @staticmethod 
    def SelectShape(threshold,min_value, max_value, feature = "width"): 
    
        number_object,labels , stats, _ = cv2.connectedComponentsWithStats(threshold, 4, cv2.CV_32S)
        
        list_width = list(stats[1: ,2])
        list_height = list(stats[1: ,3])
        list_area = list(stats[1: ,4])
        
        #
        result = np.zeros((threshold.shape[0], threshold.shape[1]),dtype =np.uint8)
        # 
        list_output_width = []
        list_output_height = []
        list_output_area =[]
        #
        list_region  = []
        # ----------------------------------------------------- WIDTH
        if feature == "width":
            for idx in range(1,number_object):
                width  = list_width[idx -1]
                height = list_height[idx -1]
                area   = list_area[idx -1]
                
                if width > min_value  and width < max_value:
                    select_object = np.zeros( (threshold.shape[0], threshold.shape[1]),dtype =np.uint8)
                    
                    select_object[labels == idx] = 255
                    select_object[labels != idx] = 0
                    
                    list_region.append(select_object)
                    result = cv2.bitwise_or(result,select_object)
                    list_output_width.append(width)
                    
                    list_output_height.append(height)
                    list_output_area.append(area)
                    
            return result,list_region,(list_output_width,list_output_height,list_output_area)
        
        # ----------------------------------------------------- HEIGHT 
        elif feature == "height":
            for idx in range(1,number_object):
                width  = list_width[idx -1]
                height = list_height[idx -1]
                area   = list_area[idx -1]
                
                if height > min_value  and height < max_value:
                    select_object = np.zeros( (threshold.shape[0], threshold.shape[1]),dtype =np.uint8)
                    
                    select_object[labels == idx] = 255
                    select_object[labels != idx] = 0
                    
                    list_region.append(select_object)
                    
                    result = cv2.bitwise_or(result,select_object)
                    list_output_width.append(width)
                    
                    list_output_height.append(height)
                    list_output_area.append(area)
                    
            return result,list_region,(list_output_width,list_output_height , list_output_area     )         
        
        # ----------------------------------------------------- AREA
        elif feature == "area":
            for idx in range(1,number_object):
                width  = list_width[idx -1]
                height = list_height[idx -1]
                area   = list_area[idx -1]
                
                if area > min_value  and area < max_value:
                    
                    select_object = np.zeros( (threshold.shape[0], threshold.shape[1]),dtype =np.uint8)
                    
                    select_object[labels == idx] = 255
                    select_object[labels != idx] = 0
                    
                    list_region.append(select_object)
                    
                    result = cv2.bitwise_or(result,select_object)
                    list_output_width.append(width)
                    
                    list_output_height.append(height)
                    list_output_area.append(area)
                    
            return result,list_region,(list_output_width,list_output_height , list_output_area)
        
        # ----------------------------------------------------- Circularity 
        # ----------------------------------------------------- Squaredness 

    @staticmethod 
    def ErodeRectangle(region , width, height):
        mask = cv2.getStructuringElement(cv2.MORPH_RECT, ksize = (2*width + 1, 2*height + 1))
        erosion = cv2.erode(region, mask , iterations = 1, borderType = cv2.BORDER_DEFAULT, borderValue = 0)
        return erosion 
    
    @staticmethod 
    def DilationRectangle(region , width, height):
        mask = cv2.getStructuringElement(cv2.MORPH_RECT, ksize = (2*width + 1, 2*height + 1))
        erosion = cv2.dilate(region, mask , iterations = 1, borderType = cv2.BORDER_DEFAULT, borderValue = 0)
        return erosion 
    
    @staticmethod 
    def drawContourMask(img, cnt):
        mask = np.zeros((img.shape[0], img.shape[1]), dtype = np.uint8)
        cv2.drawContours(mask,cnt, -1,  (255),2)
        
        return mask


class Contour():
    @staticmethod 
    def find_formula_line(point1, point2):
        # song song voi truc Ox thi y = nhau
        x1,y1 = point1
        x2,y2 = point2

        if y1 == y2:
            A = 0
            B = 1
            C = - y1

        # song song voi truc Oy thi x = nhau
        elif x1 == x2:
            A = 1
            B = 0
            C = - x1
        
        # cat ox, oy   
        else:
            A = (y1 - y2)/(x1-x2)
            B = -1
            C = y2 - A * x2

        return A, B, C
    
    @staticmethod
    def line_intersection(line1,line2):
        # Tính giá trị của x và y tương ứng với điểm giao điểm của hai đường thẳng
        # Giải hệ phương trình A1x + B1y + C1 = 0 và A2x + B2y + C2 = 0
        # Tính định thức D
        # Unboxing
        A1, B1, C1 = line1
        A2, B2, C2 = line2

        # Tính định thức hệ số x:
        det_x = B1 * C2 - B2 * C1

        # Tính định thức hệ số y:
        det_y = A2 * C1 - A1 * C2

        # Tính định thức chung:
        det = A1 * B2 - A2 * B1

        # Tính tọa độ giao điểm (x, y)
        x = det_x / det
        y = det_y / det

        return x, y
    
    @staticmethod 
    def ConvexHull(contours): 
        """ 
        contours: list of (ndarray)
        
        """
        len(contours)
        if len(contours) ==0 :
            return []
        # ConvextHull 
        hull_list = []
        for cnt in range(len(contours)):
            hull_list.append (cv2.convexHull(cnt[1:]))

        return  hull_list

    @staticmethod
    def SelectContour(contours, min_value = 0, max_value = 0, factor =0.01, feature = "width"):
        """
            Features:
                1. width
                2. height
                3. area
                4. length
                5. circularity
                6. alpha
                7. approximate
                8. anisometry
        
        """
        contour_selected = []
        
        list_feature = []
        for cnt in contours:
            
            if feature == "width" :
                _,_, width, height = cv2.boundingRect(cnt)
                if  width >= min_value and width <= max_value:
                    contour_selected.append(cnt)
                    list_feature.append(width)
                    
            elif feature == "height" :
                _,_, width, height = cv2.boundingRect(cnt)
                if  height >= min_value and height <= max_value:
                    contour_selected.append(cnt)
                    list_feature.append(height)
                    
            elif feature == "area":
                area = cv2.contourArea(cnt)
                if  area >= min_value and area <= max_value:
                    contour_selected.append(cnt)
                    list_feature.append(area)
                    
            elif feature == "length": # P 
                per = cv2.arcLength(cnt,True)
                if  per >= min_value and per <= max_value:
                    contour_selected.append(cnt)
                    list_feature.append(per)
                    
            elif feature == "circularity":
                dis = []
                area = cv2.contourArea(cnt)
                
                moment = cv2.moments(cnt)
                center_x = moment["m10"] / moment["m00"]
                center_y = moment["m01"] / moment["m00"]
                
                for point in cnt: 
                    x_codinate = point[0][0]
                    y_codinate = point[0][1]
                    d = math.sqrt((x_codinate - center_x)**2 + (center_y - y_codinate)**2)
                    dis.append(d)
                        
                max_dis = max(dis)
                cicularity  = area / (math.pi * max_dis**2)
                if  cicularity >= min_value and cicularity <= max_value:
                    contour_selected.append(cnt)
                    list_feature.append(cicularity)
                
            elif feature == "alpha": # Hinh chu nhat nghieng bao nhieu do so voi truc 0Y
                rect = cv2.minAreaRect(cnt)
                alpha = rect[2]
                if  alpha >= min_value and alpha <= max_value:
                    contour_selected.append(cnt)
                    list_feature.append(alpha)
                    
            elif feature == "approximate":
                epsilon = factor * cv2.arcLength(cnt,True)
                approx = cv2.approxPolyDP(cnt,epsilon,True)
                
                if len(approx) >= min_value and len(approx) <= max_value:
                    contour_selected.append(approx)
                    list_feature.append(len(approx))
                    
            elif feature == "anisometry":
                moment = cv2.moments(cnt)
                
                center_x = moment["m10"] / moment["m00"]
                center_y = moment["m01"] / moment["m00"]
                
                def calc_moment(contours):
                    m20 = 0 
                    m02 = 0 
                    m11 = 0 
            
                    for point in contours: 
                        x_codinate = point[0][0]
                        y_codinate = point[0][1]
                        
                        m20 += (x_codinate - center_x) **2 
                        m02 += (y_codinate - center_y) **2 
                        
                        m11 = (x_codinate - center_x) * (y_codinate - center_y)
                    return m20 , m02 , m11

                def calc_anisometry (m20 , m02 , m11 ):
                    R_a = math.sqrt(8* (m20 + m02 + math.sqrt((m20 -m02 )**2 + 4 * m11**2))) / 2 
                    R_b = math.sqrt(8* (m20 + m02 - math.sqrt((m20 -m02 )**2 + 4 * m11**2))) / 2
                    
                    return R_a / R_b
                
                m20 , m02 , m11 = calc_moment(cnt)
                anisometry = calc_anisometry(m20 , m02 , m11)
                if anisometry >= min_value and anisometry <= max_value:
                    contour_selected.append(cnt)
                    list_feature.append(anisometry)
                
                                
                                
        print(f'features - {feature}: {list_feature}')          
        return contour_selected     

    @staticmethod
    def find_perpendicular_line(p1, line):
        if len(p1) !=2 or len(line)!=3:
            return (0,0,0)
        
        x1, y1 = p1
        A, B, C = line
        return (-B,A, B*x1 -A*y1 )
    
    @staticmethod
    def distance_two_point(point1, point2):
        if len(point1)!=2 or len(point2)!=2:
            return None
        
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x1-x2)**2 + (y1-y2)**2)
    
    @staticmethod 
    def minimum_distance_point_to_line(point, line):
        if len(point)!=2 or len(line)!=3:
            return None
        
        x1,y1 = point
        A, B, C  = line
        if A==0 and B==0:
            return None
        
        dis_min = abs(A*x1 + B*y1 +C) /  math.sqrt(A**2 + B**2)
        return dis_min
        
class Distance():
    pass

class Image():

    @staticmethod 
    @numba.jit(nopython = True,cache = True, parallel = True)
    
    def emphasize(img, kernel_size = 3, c = 1.0):
        
        
        height, width = img.shape[0], img.shape[1]
        
        emphasize_space  = np.zeros((img.shape[0], img.shape[1]),dtype = np.float64)
        k_height, k_width  = kernel_size ,kernel_size

        for y in numba.prange(height - k_height ):
            for x in numba.prange(width - k_width):
                
                crop  = img[y : y + k_height , x: x + k_width]
                
                emphasize_space[y : y + k_height ,x: x + k_width] = (crop - np.mean(crop))*c + crop
                
        np.clip(emphasize_space, 0, 255, out=emphasize_space)
        
        return emphasize_space.astype(np.uint8) 