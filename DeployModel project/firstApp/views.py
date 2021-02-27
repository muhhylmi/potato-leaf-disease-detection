from django.shortcuts import render

from django.core.files.storage import FileSystemStorage
from django.conf import settings 
from django.conf.urls.static import static


import cv2
import numpy as np
import math
import pandas as pd
from pandas import DataFrame

from skimage import data
from skimage.filters import threshold_multiotsu
import joblib
from IPython.display import HTML 
import json 


from skimage import data
from skimage.filters import threshold_multiotsu

def ml(data):
    def grabcut(img):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = np.zeros(image.shape[:2],np.uint8)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        rect = (30,30,190,190)
        cv2.grabCut(image,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        image = image*mask2[:,:,np.newaxis]
        return image

    def extractdisease(img):
        #ekstrak bercak
        lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
        L,A,B=cv2.split(lab)
        thresholds = threshold_multiotsu(A)
        # Using the threshold values, we generate the three regions.
        regions = np.digitize(A, bins=thresholds)
        T=[]
        for thresh in thresholds:
            T.append(thresh)
            def th(array):
                temp_img = np.copy(array)
                for i in range(len(temp_img)):
                    for j in range(len(temp_img[0])):
                        if(temp_img[i][j] >= T[1] and temp_img[i][j] >= T[0]):
                            temp_img[i][j] = 255
                        else:
                            temp_img[i][j] = 0
                return temp_img
        th = th(A)
        mask = th
        result = cv2.bitwise_and(img, img, mask=mask)
        return result

    def morf(result1):
         #opening closing untuk fitur bentuk
        kernel1 = kernel = np.ones((5,5),np.uint8)
        r_opening = cv2.morphologyEx(result1, cv2.MORPH_OPEN, kernel1)
        r_closing = cv2.morphologyEx(r_opening, cv2.MORPH_CLOSE, kernel1)
        #EKSTRAKSI FITUR BENTUK
        res_gray = cv2.cvtColor(result1, cv2.COLOR_RGB2GRAY)
        res_gray2 = cv2.cvtColor(r_closing, cv2.COLOR_RGB2GRAY)
        ret, thresh = cv2.threshold(res_gray2, 0, 255, cv2.THRESH_BINARY)
        return thresh


    def area(img):
        citraBerwarna = cv2.merge((img, img, img))
        kontur, hirarki = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        luas = 0
        l=0
        terbesar =0
        for i in range(len(kontur)):
            if len(kontur)==0:
                l=0
            else:        
                luas = cv2.contourArea(kontur[i])
                if i == 0:
                    posisi = 0
                    terbesar = luas
                else :
                    if luas > terbesar:
                        posisi = i
                        terbesar = luas
                        luas = terbesar
        for i in range(len(kontur)):
            if i != posisi:
                cv2.drawContours(citraBerwarna,kontur,
                                 i, (0,0,0), -1)
        l = terbesar
        return l

    #perimeter
    def perimeter (img): 
        citraBerwarna = cv2.merge((img, img, img))
        kontur, hirarki = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        luas = 0
        posisi = 0
        p = 0
        terbesar =0
        terbesar = 0
        for i in range(len(kontur)):
            luas = cv2.arcLength(kontur[i], True)
            if i == 0:
                posisi = 0
                terbesar = luas
            else :
                if luas > terbesar:
                    posisi = i
                    terbesar = luas
                    luas = terbesar
        for i in range(len(kontur)):
            if i != posisi:
                cv2.drawContours(citraBerwarna,kontur,
                                i, (0,0,0), -1)
        p = terbesar
        return p

    def solidity(img):
        l = area(img)
        citraBerwarna = cv2.merge((img, img, img))
        kontur, hirarki = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        posisi = 0
        terbesar = 0
        for i in range(len(kontur)):
            luas = cv2.contourArea(kontur[i])
            if i == 0:
                posisi = 0
                terbesar = luas
            else :
                if luas > terbesar:
                    posisi = i
                    terbesar = luas
                    luas = terbesar
        if len(kontur) == 0:
            convex = 0
            l_hull =0
            res = 0
        else:
            convex = cv2.convexHull(kontur[posisi])  
            l_hull = cv2.contourArea(convex)
            res = l/l_hull
        return res

    def compactness(img):
        phi = 3.14
        are = area(img)
        perimete = perimeter(img)
        if are == 0:    
            res=0
        else:
            res = 4*phi*are/(np.power(perimete,2))
        return res

    def convexity(img):
        perimete = perimeter(img)
        citraBerwarna = cv2.merge((img, img, img))
        kontur, hirarki = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        posisi = 0
        i=0
        terbesar =0
        for i in range(len(kontur)):
            luas = cv2.arcLength(kontur[i],True)
            if i == 0:
                posisi = 0
                terbesar = luas
            else :
                if luas > terbesar:
                    posisi = i
                    terbesar = luas
                    luas = terbesar
        if perimete == 0:
            res =0
        else:
            if posisi == i+1:
                cp = cv2.arcLength(cv2.convexHull(kontur[posisi-1]), True)
            else:
                cp = cv2.arcLength(cv2.convexHull(kontur[posisi]), True)
                res = cp/perimete 
        return res

     #eKSTRAKSI FITUR TEKSTUR
    def contrast(matrix):
        w, h = matrix.shape
        res = 0
        for i in range(w):
            for j in range(h):
                res+= matrix[i][j] * np.power(i-j, 2)
        return res

    def energy(matrix):
        w,h = matrix.shape
        res = 0
        for i in range(w):
            for j in range(h):
                res += np.power(matrix[i][j], 2)
        return res

    def homogenity(matrix):
        w, h = matrix.shape
        res = 0 
        for i in range(w):
            for j in range(h):
                res += matrix[i][j] / (1 + np.power(i-j, 2))
        return res

    def correlation(matrix):
        w, h = matrix.shape
        res = 0
        jml =0 
        stdv=0
        for i in range(w):
            for j in range(h):
                jml += matrix[i][j]             
        rata = jml/(w*h)
        for i in range(w):
            for j in range(h):
                stdv += np.power((matrix[i][j] - rata),2)
        standar_deviasi = math.sqrt(stdv/(w*h))
        for i in range(w):
            for j in range(h):
                res += (((i-rata)*(j-rata)*matrix[i][j])/(np.power(standar_deviasi,2)))
        return res

    def entropy(matrix):
        w, h = matrix.shape
        res = 0
        for i in range(w):
            for j in range(h):
                if matrix[i][j] > 0:
                    res += matrix[i][j] * np.log2(matrix[i][j])
        return res

    def glcm(imag, degree):
        scale_percent = 60 # percent of original size
        width = int(imag.shape[1] * scale_percent / 100)
        height = int(imag.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(imag, dim, interpolation = cv2.INTER_AREA)
        arr = np.array(resized)
        res = np.zeros((arr.max() + 1, arr.max()+1), dtype=int)
        w,h=arr.shape
        if degree == 0:
            for i in range(w-1):
                for j in range (h-1):
                    if arr[j, i+1] and arr[j, i] == 0:
                        res[arr[j, i+1], arr[j, i]] += 0
                    else:
                        res[arr[j, i+1], arr[j, i]] += 1
        elif degree == 45:
            for i in range(w-1):
                for j in range(h-1):
                    res[arr[j-1, i+1], arr[j, i]] += 1
        elif degree == 90:
            for i in range(w-1):
                for j in range(h-1):
                    res[arr[j-1, i], arr[j, i]] += 1
        elif degree == 135:
            for i in range(w-1):
                for j in range(h-1):
                    res[arr[j-1, i-1], arr[j,i]] += 1
        else:
            print('sudut tidak ada')
        return res

        #EKSTRAKSI FITUR WARNA
    def mean(img):
        b,g,r = cv2.split(img)
        rataB = np.mean(b, axis=(0,1))
        rataG = np.mean(g, axis=(0,1))
        rataR = np.mean(r, axis=(0,1))
        res = (rataB + rataG + rataR)/3
        return res

    def st_deviation(img):
        b,g,r = cv2.split(img)   
        jumlah = b+g+r
        arr = np.array(jumlah)
        w,h = arr.shape
        rata = mean(img)
        var = 0
        for i in range(w-1):
            for j in range(h-1):
                var += np.power((arr[i][j]-rata),2)
        res = math.sqrt(var/(w*h))
        return res

    def skewness(img):
        n=3
        s_deviasi = st_deviation(img)
        b,g,r = cv2.split(img)   
        jumlah = b+g+r
        arr = np.array(jumlah)
        w,h = arr.shape
        rata = mean(img)
        var = 0
        if s_deviasi == 0:
            res =0    
        else:     
            for i in range(w-1):
                for j in range(h-1):
                    var += np.power((arr[i][j]-rata),n)
            res = var/(w*h*(np.power(s_deviasi,n)))
        return res

    def kurtosis(img):
        n=4
        s_deviasi = st_deviation(img)
        b,g,r = cv2.split(img)   
        jumlah = b+g+r
        arr = np.array(jumlah)
        w,h = arr.shape
        rata = mean(img)
        var = 0
        if s_deviasi==0:
            res=0
        else:            
            for i in range(w-1):
                for j in range(h-1):
                    var += np.power((arr[i][j]-rata),n)
            res = var/(w*h*(np.power(s_deviasi,n)))
        return res

    def eks_fitur_contrast(imge):
        gl_0 = glcm(imge,0)
        return (contrast(gl_0))

    def eks_fitur_energy(imge):
        gl_0 = glcm(imge,0)
        return (energy(gl_0))

    def eks_fitur_homogenity(imge):
        gl_0 = glcm(imge,0)
        return (homogenity(gl_0))

    def eks_fitur_correlation(imge):
        gl_0 = glcm(imge,0)
        return (correlation(gl_0))

    def eks_fitur_mean(imge):    
        return mean(imge)

    def eks_fitur_deviation(imge):
        return st_deviation(imge)

    def eks_fitur_skewness(imge):
        return skewness(imge)

    def eks_fitur_kurtosis(imge):
        return kurtosis(imge)

    def eks_fitur_area(imge):
        return area(imge)

    def eks_fitur_perimeter(imge):
        return perimeter(imge)

    def eks_fitur_solidity(imge):
        return solidity(imge)

    def eks_fitur_compactness(imge):
        return compactness(imge)

    def eks_fitur_convexity(imge):
        return convexity(imge)


    d_contras=[]; d_energi= []; d_homogeniti= []
    d_correlation= []; d_mean = []; d_st_deviation=[]
    d_skewness =[]; d_kurtosis =[] ; d_perimeter =[] 
    d_area = []; d_solidity =[]; d_compactness =[]
    d_convexity = []

    img = cv2.imread(data)
    grabcut(img)
    res = grabcut(img)
    extractdisease(res)
    res2 = extractdisease(res)
    res_gray = cv2.cvtColor(res2, cv2.COLOR_RGB2GRAY)
    morf(res2)
    thresh = morf(res2)

    #PENGAMBILAN NILAI
    eks_fitur_contrast(res_gray)
    d_contras.append(eks_fitur_contrast(res_gray))

    eks_fitur_energy(res_gray)
    d_energi.append(eks_fitur_energy(res_gray))

    eks_fitur_homogenity(res_gray)
    d_homogeniti.append(eks_fitur_homogenity(res_gray))

    correlation(glcm(res_gray,0))
    d_correlation.append(eks_fitur_correlation(res_gray))

    eks_fitur_mean(res)
    d_mean.append(eks_fitur_mean(res))

    eks_fitur_deviation(res)
    d_st_deviation.append(eks_fitur_deviation(res))

    eks_fitur_skewness(res)
    d_skewness.append(eks_fitur_skewness(res))

    eks_fitur_kurtosis(res)
    d_kurtosis.append(eks_fitur_kurtosis(res))

    eks_fitur_area(thresh)
    d_area.append(eks_fitur_area(thresh))

    eks_fitur_perimeter(thresh)
    d_perimeter.append(eks_fitur_perimeter(thresh))    

    eks_fitur_solidity(thresh)
    d_solidity.append(eks_fitur_solidity(thresh))

    eks_fitur_compactness(thresh)
    d_compactness.append(eks_fitur_compactness(thresh))

    eks_fitur_convexity(thresh)
    d_convexity.append(eks_fitur_convexity(thresh))

    data = {'Contrast':d_contras, 'Energy':d_energi, 'Homogenity':d_homogeniti,'Correlation':d_correlation,
           'Mean':d_mean, 'Deviasi':d_st_deviation, 'Skewness':d_skewness, 'Kurtosis': d_kurtosis,
           'Area':d_area, 'Perimeter':d_perimeter, 'Solidity':d_solidity, 'Compactness':d_compactness,'Convexity':d_convexity}
    data = pd.DataFrame(data)

    return data



# Create your views here.
def index(request):

	context={'a':1}
	return render(request,'index.html', context)


def dataset(request):
        df = pd.read_excel (r'D:\KULIAH\Tugas Kuliah\Semester 8\Skripsi\dataset\datadenganyangbaru.xlsx')
        # df = df.to_html(classes='table')
        json_records = df.reset_index().to_json(orient ='records')
        data=[]
        data = json.loads(json_records)

        context={'dataset':data}
        return render(request,'data.html', context)


def prediksi(request):
	context={'a':1}
	return render(request,'prediksi.html', context)	


def showFitur(data):
    # datahtml = data.to_html(classes='table')
    json_records = data.reset_index().to_json(orient ='records')
    data=[]
    data = json.loads(json_records)
    return data

def predictImage(request):
        import time
        tic = time.perf_counter()
        fileObj = request.FILES['filepath']
        fs = FileSystemStorage()
        filePathName = fs.save(fileObj.name,fileObj)
        filePathName = fs.url(filePathName)
        url = '.'+ filePathName
        data = ml(url.replace("%20", " "))
        df = pd.read_excel (r'D:\KULIAH\Tugas Kuliah\Semester 8\Skripsi\dataset\datadenganyangbaru.xlsx')
        X=df.drop(['Target'], axis=1)
        X_min = X.min()
        X_max = X.max()
        X_range = (X_max - X_min)
        data_scaled = (data - X_min)/(X_range)
        data_html = showFitur(data_scaled)

        joblib_file = "./models/SVM_model_linear.pkl" 
        joblib_model = joblib.load(joblib_file)
        y = joblib_model.predict(data_scaled)
        prediksi = ''
        if(y==0):
            prediksi = 'early blight'
        elif(y==1):
            prediksi = 'late blight'
        else:
            prediksi = 'daun sehat'
        toc = time.perf_counter()

        time = toc - tic

        context={'filePathName':filePathName, 'prediksi':prediksi, 'dataset':data_html, 'waktu':"{:.2f}".format(time)}
        return render(request,'prediksi.html', context)
	
from django.core.paginator import Paginator

from django import template

register = template.Library()

@register.filter
def multiply(value, arg):
    return value * arg

def dataImage(request):
    data =[]
    for x in range(1,451):
        data.append("/static/data/a ("+str(x)+").JPG")
    paginator = Paginator(data, 15)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    if page_number is None:
        no = 0
    else:
        no = int(page_number)*15-15

    context={'data':page_obj , 'no':no, 'page':page_number}
    return render(request,'dataImage.html', context)