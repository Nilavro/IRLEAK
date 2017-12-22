# In[1]:

import pymysql
import numpy as np
import math
import cv2
from os import listdir
import matplotlib.pyplot as plt
from collections import namedtuple

import humandetection
import IRLeakStitch as st

NUM_PICTURES = 64

import graphviz
from multiprocessing import Pool, Lock, Queue, Pipe, Process
import sys
NUM_PROCESSES = 12
#from IPython.display import display

#get_ipython().magic('matplotlib notebook')


# In[2]:

my_conn=lambda:pymysql.connect(host='127.0.0.1',user='david',password='w4vrds9nse',db='irleak',charset='utf8mb4',cursorclass=pymysql.cursors.DictCursor)


# In[3]:

theta_l = 65 * math.pi / 180
f = 1 / math.tan(theta_l / 2)
f_pix = 360 * f # the 360 is pixels, half img width, not degrees
f_pix


# In[4]:

theta_r = lambda: ((513 - 1) / NUM_PICTURES) * 2 * math.pi / 513
delta_col = lambda: 2*f_pix*math.sin(theta_r()/2)


# In[5]:

s = f_pix
x2xp = lambda x: s * math.atan(x / f_pix)
xp2x = lambda xp: f_pix * math.tan(xp / s)
y2yp = lambda x, y: s * y / math.sqrt((x**2) + (f_pix**2))
yp2y = lambda xp, yp: f_pix * yp * (1/math.cos(xp / s)) / s


# In[6]:

def parse_ir(f_name):
    out = list()
    vals = list()
    with open(f_name) as FILE:
        vals = [float(n) for n in FILE.read().strip().split(',')]
    for i in range(0, 64, 4):
        out.append(vals[i:i+4])
    out = np.array(out)
    #out = np.fliplr(out)
    out = np.flipud(out)
    return out


def ir_image(im, max_temp, min_temp):
    therm_im = np.zeros(list(im.shape)+[3],dtype=np.uint8)
    mid_temp = (max_temp + min_temp)/2
    rng_temp = max_temp - min_temp
    #[H, S, V]
    therm_im[:,:,0] = (170 * (max_temp - im) / rng_temp) 
    therm_im[:,:,1] = 255
    therm_im[:,:,2] = 127
    return cv2.cvtColor(therm_im, cv2.COLOR_HSV2BGR)


def gs_image(im, max_temp, min_temp):
    rng_temp = max_temp-min_temp
    therm_im = np.zeros(im.shape, dtype=np.uint8)
    therm_im = 255*(im-min_temp)//rng_temp
    return therm_im


def reconvertTemp(thPan, max_temp, min_temp):
    return ((max_temp-min_temp) * thPan / 255) + min_temp


def alignment_matrix():
    mats = []
    
    # (x, y)
    src = np.array([
        (0, 0),
        (2, 10),
        (1, 11),
        (3, 5)
    ], dtype = np.float32)
    dst = np.array([
        (171, 713),
        (324, 239),
        (194, 209),
        (380, 445)
    ], dtype = np.float32)
    mats.append(cv2.getPerspectiveTransform(src, dst))
    
    src = np.array([
        (1, 8),
        (2, 4),
        (3, 12),
        (0, 7)
    ], dtype = np.float32)
    dst = np.array([
        (257, 311),
        (340, 490),
        (372, 165),
        (208, 356)
    ], dtype = np.float32)
    mats.append(cv2.getPerspectiveTransform(src, dst))
    
    src = np.array([
        (3, 12),
        (3, 7),
        (0, 4),
        (2, 8)
    ], dtype = np.float32)
    dst = np.array([
        (380, 160),
        (336, 386),
        (100, 500),
        (279, 352)
    ], dtype = np.float32)
    mats.append(cv2.getPerspectiveTransform(src, dst))
    
    return sum(mats)/len(mats)


mat = alignment_matrix()


# In[7]:

#sets = ('/home/david/tank/IRLeakData/' + d + '/' for d in listdir('/home/david/tank/IRLeakData/') if (d[:3]=='03-' and int(d[3:]) > 1497025035259))
#sets = ('/home/david/remote/IRLeakData/' + d + '/' for d in listdir('/home/david/remote/IRLeakData/') if (d[:3]=='03-' and int(d[3:]) > 1497025035259))
sets = ('/home/david/eclipse/IRLeakData/' + d + '/' for d in listdir('/home/david/eclipse/IRLeakData/') if (d[:3]=='03-' and int(d[3:]) > 1497025035259))
#sets = ('/home/david/Dump/IRLeakData/' + d + '/' for d in listdir('/home/david/Dump/IRLeakData/') if (d[:3]=='03-' and int(d[3:]) > 1497025035259))
#alert_ims = []
#for setname in list(sets)[:6]:
#for setname in sorted(list(sets)):

Alert = namedtuple('Alert', ['setname','imgs','descs','pano', 'persons'])

def process_set(setname):#, _q):
    IRname  = setname + 'temperatures/'
    dirname = setname + 'rotated/'
    
    try:
        th_ims = [parse_ir(IRname + im) for im in sorted(listdir(IRname))]
    except FileNotFoundError:
        return
    max_temp = np.amax(th_ims)
    min_temp = np.amin(th_ims)
    #print(min_temp, max_temp)
    #ir_ims = [ir_image(im, max_temp, min_temp) for im in ir_ims]
    ir_ims = [ir_image(im, max_temp, min_temp) for im in th_ims]
    th_ims = [gs_image(im, max_temp, min_temp) for im in th_ims]
    
    w_ims = [cv2.warpPerspective(im, mat, (480, 720)) for im in ir_ims]
    w_ths = [cv2.warpPerspective(im, mat, (480, 720)) for im in th_ims]
    
    ims = [cv2.imread(dirname + img) for img in sorted(listdir(dirname)) if img[-4:] == '.jpg']
    alpha = 0.5
    try:
        o_ims = [cv2.addWeighted(ims[i],alpha,w_ims[i],1-alpha,0) for i in range(NUM_PICTURES)]
    except IndexError:
        #print("not enough images", len(ims), len(w_ims))
        return
    cyl_ims = []
    cyl_ths = []
    (lb,rb) = (13,467)

    center = (ims[0].shape[1]/2, ims[0].shape[0]/2)
    mat_x = np.zeros(ims[0].shape[:2], np.float32)
    mat_y = np.zeros(ims[0].shape[:2], np.float32)
    for row in range(ims[0].shape[0]):
        for col in range(ims[0].shape[1]):
            x = col - center[0]
            y = row - center[1]
            mat_y[row, col] = yp2y(x, y) + center[1]
            mat_x[row, col] = xp2x(x) + center[0]

    cyl_vis = [cv2.remap(im, mat_x, mat_y, cv2.INTER_LINEAR)[145:,195:377] for i,im in enumerate(ims)]

    for i,im in enumerate(o_ims):
        #reim = cv2.remap(im, mat_x, mat_y, cv2.INTER_LINEAR)[:,lb:rb]
        reim = cv2.remap(im, mat_x, mat_y, cv2.INTER_LINEAR)[145:, 195:377]
        cyl_ims.append(reim)
    for i,im in enumerate(w_ths):
        reim = cv2.remap(im, mat_x, mat_y, cv2.INTER_LINEAR)[145:, 195:377]
        cyl_ths.append(reim)
    
    N_ims = NUM_PICTURES

    s_ims = []
    s_ths = []
    s_vis = []
    for i in range(N_ims):
        im = np.zeros((cyl_ims[i].shape[0],cyl_ims[i].shape[1]+int(delta_col()*(N_ims-1)),cyl_ims[i].shape[2]),np.uint8)
        th = np.zeros(im.shape[:2], np.uint8)
        vi = np.zeros(im.shape, np.uint8)
        im[:cyl_ims[i].shape[0],int(delta_col()*i):int(delta_col()*i)+cyl_ims[i].shape[1]] = cyl_ims[i]
        th[:cyl_ims[i].shape[0],int(delta_col()*i):int(delta_col()*i)+cyl_ims[i].shape[1]] = cyl_ths[i]
        vi[:cyl_ims[i].shape[0],int(delta_col()*i):int(delta_col()*i)+cyl_ims[i].shape[1]] = cyl_vis[i]
        s_ims.append(im)
        s_ths.append(th)
        s_vis.append(vi)

    gpIms = []
    gpThs = []
    gpVis = []
    for i,im in enumerate(s_ims):
        G = im.copy()
        gpIms.append([G])
        Gth = s_ths[i].copy()
        gpThs.append([Gth])
        Gvi = s_vis[i].copy()
        gpVis.append([Gvi])
        for j in range(6):
            G = cv2.pyrDown(G)
            Gth = cv2.pyrDown(Gth)
            Gvi = cv2.pyrDown(Gvi)
            gpIms[i].append(G)
            gpThs[i].append(Gth)
            gpVis[i].append(Gvi)

    lpIms = []
    lpThs = []
    lpVis = []
    for i,gpIm in enumerate(gpIms):
        lpIms.append([gpIm[5]])
        lpThs.append([gpThs[i][5]])
        lpVis.append([gpVis[i][5]])
        for j in range(5,0,-1):
            GE = cv2.pyrUp(gpIm[j])
            GEth = cv2.pyrUp(gpThs[i][j])
            GEvi = cv2.pyrUp(gpVis[i][j])
            hmax = min(GE.shape[0], gpIm[j-1].shape[0])
            wmax = min(GE.shape[1], gpIm[j-1].shape[1])
            if i%2 == 0:
                L = cv2.subtract(gpIm[j-1][:hmax, :wmax], GE[:hmax, :wmax])
                Lth = cv2.subtract(gpThs[i][j-1][:hmax, :wmax], GEth[:hmax, :wmax])
                Lvi = cv2.subtract(gpVis[i][j-1][:hmax, :wmax], GEvi[:hmax, :wmax])
            else:
                L = cv2.subtract(gpIm[j-1][-hmax:, -wmax:], GE[-hmax:, -wmax:])
                Lth = cv2.subtract(gpThs[i][j-1][-hmax:, -wmax:], GEth[-hmax:, -wmax:])
                Lvi = cv2.subtract(gpVis[i][j-1][-hmax:, -wmax:], GEvi[-hmax:, -wmax:])
            lpIms[i].append(L)
            lpThs[i].append(Lth)
            lpVis[i].append(Lvi)
    
    LS = []
    for lx in zip(*lpIms):
        cols = lx[0].shape[1]
        part = lambda x: (x*cols)//N_ims
        ls = np.hstack((lx[i][:,part(i):part(i+1)] for i in range(N_ims)))
        LS.append(ls)
    
    TH = []
    for tx in zip(*lpThs):
        cols = tx[0].shape[1]
        part = lambda x: (x*cols)//N_ims
        ts = np.hstack((tx[i][:,part(i):part(i+1)] for i in range(N_ims)))
        TH.append(ts)
    
    VI = []
    for vx in zip(*lpVis):
        cols = vx[0].shape[1]
        part = lambda x: (x*cols)//N_ims
        vs = np.hstack((vx[i][:,part(i):part(i+1)] for i in range(N_ims)))
        VI.append(vs)
    
    ls_ = LS[0]
    ts_ = TH[0]
    vs_ = VI[0]
    for i in range(1,6):
        ls_ = cv2.pyrUp(ls_)
        ts_ = cv2.pyrUp(ts_)
        vs_ = cv2.pyrUp(vs_)
        hmax = min(ls_.shape[0], LS[i].shape[0])
        wmax = min(ls_.shape[1], LS[i].shape[1])
        ls_ = cv2.add(ls_[:hmax, :wmax], LS[i][:hmax, :wmax])
        ts_ = cv2.add(ts_[:hmax, :wmax], TH[i][:hmax, :wmax])
        vs_ = cv2.add(vs_[:hmax, :wmax], VI[i][:hmax, :wmax])
    
    cols = s_ims[0].shape[1]
    part = lambda x: (x*cols)//N_ims
    simpleOver = np.hstack((s_ims[i][:,part(i):part(i+1)] for i in range(N_ims)))
    simpleTemp = np.hstack((s_ths[i][:,part(i):part(i+1)] for i in range(N_ims)))
    simpleVisu = np.hstack((s_vis[i][:,part(i):part(i+1)] for i in range(N_ims)))

    #
    #
    # TENSORFLOW HERE!!!
    #num_persons = humandetection.count_persons(vs_)
    #_q.put(vs_)
    #num_persons = _q.get()
    #has_person = num_persons > 0
    #print(setname, num_persons)
    
    temps = reconvertTemp(ts_, max_temp, min_temp)
    
    ret, thresh = cv2.threshold(ts_,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    HVAC_SET = None
    #HVAC_SET = 70
    if HVAC_SET is None:
        try:
            ts = float(setname[-14:-1])/1000
            with my_conn() as CONN:
                with CONN.cursor() as cur:
                    cur.execute('select timestamp, target ' + 
                                'from nest_data where timestamp<%s ' +
                                'order by timestamp desc limit 1', (ts,))
                    for res in cur:
                        HVAC_SET = float(res['target'])
                CONN.commit()
        except:
            pass
    
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    sure_bg = 255 - sure_bg
    
    ret, markersA = cv2.connectedComponents(sure_bg)
    markersA = markersA+1
    markersA[unknown==255] = 0
    markersA = cv2.watershed(vs_, markersA)
    #markersA = cv2.watershed(cv2.cvtColor(ts_, cv2.COLOR_GRAY2BGR), markersA)
    
    alert_ims = []
    mean = np.mean(temps)
    stdd = np.std(temps)
    imgAlert = vs_.copy()
    alerthot, thermhot, alertcol, thermcol = [], [], [], []
    MIN_DIM = 250
    for g in set(markersA.flat):
        loc = np.mean(temps[markersA == g])
        zone = np.where(markersA==g)
        print(zone)
        sys.exit(0)
        (t, b, l, r) = (min(zone[0]),max(zone[0]),min(zone[1]),max(zone[1]))
        h = b-t
        w = r-l
        ctr = [h//2+b, w//2+l]
        #(h,w,t,b,l,r) = (1,1,0,1,0,1)
        #ctr = [int(sum(zone[0])/len(zone[0])), int(np.mean(zone[1]))]
        #print(ctr, h, w, t, b, l, r)
        if h < MIN_DIM:
            ctr[0] = max(MIN_DIM//2, ctr[0])
            ctr[0] = min(vs_.shape[0]-(MIN_DIM//2), ctr[0])
            #b = min(b + (100 - (h))//2, vs_.shape[0])
            b = ctr[0] + (MIN_DIM//2)
            #t = max(t - (100 - (h))//2, 0)
            t = ctr[0] - (MIN_DIM//2)
            h = b-t
        if w < MIN_DIM:
            ctr[1] = max(MIN_DIM//2, ctr[1])
            ctr[1] = min(vs_.shape[1]-(MIN_DIM//2), ctr[1])
            #r = min(r + (100 - (w))//2, vs_.shape[1])
            r = ctr[1] + (MIN_DIM//2)
            #l = max(l - (100 - (w))//2, 0)
            l = ctr[1] - (MIN_DIM//2)
            w = r-l
        #print(ctr, h, w, t, b, l, r, loc)
        if loc > mean+stdd and (HVAC_SET is None or HVAC_SET < mean):
            #print(g,loc, 'Hot!')
            imgAlert[markersA == g] = [0,0,255]
            alert_ims.append(vs_[t:b,l:r])
            #alerthot.append(cv2.imwrite('alertAhot'+str(g)+'.png', vs_[t:b,l:r]))
            #thermhot.append(cv2.imwrite('thermalertAhot'+str(g)+'.png', ts_[t:b,l:r]))
        elif loc < mean-stdd and (HVAC_SET is None or HVAC_SET > mean):
            #print(g, loc, 'Cold')
            imgAlert[markersA == g] = [255,0,0]
            alert_ims.append(vs_[t:b,l:r])
            #alertcol.append(cv2.imwrite('alertAcol'+str(g)+'.png', vs_[t:b,l:r]))
            #thermcol.append(cv2.imwrite('thermalertAcol'+str(g)+'.png', ts_[t:b,l:r]))
    alert_des = [st.detect_describe(im) for im in alert_ims]
    _ = [cv2.imwrite(setname+str(i)+'.png', im) for i,im in enumerate(alert_ims)]
    #return (alert_set, num_persons, has_person, setname, alert_des)
    return Alert(setname, alert_ims, alert_des, vs_, 0)


#sets = sorted(list(sets))
#qs = [Queue() for set in sets]
#sqs = zip(sets,qs)
#startq = Queue()
#_ = [startq.put(sq) for sq in sqs]
#for i in range(NUM_PROCESSES):
#    startq.put(None)
#endq = Queue()
#procs = [Process(target=targ, args=(startq,endq,))]
#_ = [p.start() for p in procs]
#for q in qs:
#    q.put(humandetection.count_persons(q.get()))
#_ = [p.join() for p in procs]
#alert_ims = [x for x in endq if x is not None]


#alert_ims = [x for x in map(process_set, sets) if x is not None]
#alerts = [x for x in map(process_set, sets) if x is not None]

def tf_helper(al):
    if al is None:
        return None
    else:
        count = humandetection.count_persons(al.pano)
        return Alert(*al[:4], humandetection.count_persons(al.pano))


with Pool(NUM_PROCESSES) as p:
    with Pool(1) as tfpool:
        al_futures = [p.apply_async(process_set, [setname]) for setname in sets]
        cnt_futures = [tfpool.apply_async(tf_helper, [af.get()]) for af in al_futures]
        alerts = [x for x in (fut.get() for fut in cnt_futures) if x is not None]

    

# In[8]:
g = graphviz.Graph(format='png')
#for i, a_set in enumerate(alert_ims):
for i, a_set in enumerate(alerts):
    #for j, b_set in enumerate(alert_ims[i+1:]):
    if a_set.persons > 0:
        continue
    for j, b_set in enumerate(alerts[i+1:]):
            if b_set.persons > 0:
                continue
            #for a, a_im in enumerate(a_set[0]):
            for a, a_im in enumerate(a_set.imgs):
                #cv2.imwrite(str(i)+'a.jpg', a_im)
                #(kp_a, f_a) = a_set[4][a]
                (kp_a, f_a) = a_set.descs[a]
                for b, b_im in enumerate(b_set.imgs):
                    #(kp_b, f_b) = b_set[4][b]
                    (kp_b, f_b) = b_set.descs[b]
                    M = None
                    try:
                        M = st.match_keypoints(kp_a, kp_b, f_a, f_b, .75, 4)
                    except:
                        continue
                    if M is None:
                        continue
                    if len(M[0]) < 80:
                        continue
                    #print(a_set[3][-17:-1]+'--'+str(i)+'.'+str(a), b_set[3][-17:-1]+'--'+str(i+j+1)+'.'+str(b))
                    print(a_set.setname[-17:-1]+'--'+str(i)+'.'+str(a), b_set.setname[-17:-1]+'--'+str(i+j+1)+'.'+str(b))
                    g.edge(
                        #a_set[3][-17:-1]+'--'+str(i)+'.'+str(a),
                        a_set.setname[-17:-1]+'--'+str(i)+'.'+str(a),
                        #b_set[3][-17:-1]+'--'+str(i+j+1)+'.'+str(b),
                        b_set.setname[-17:-1]+'--'+str(i+j+1)+'.'+str(b),
                        label=str(len(M[0]))
                    )
                    #cv2.imwrite(a_set[3][-17:-1]+'--'+str(i)+'.'+str(a)+'.png', a_im)
                    cv2.imwrite(a_set.setname[-17:-1]+'--'+str(i)+'.'+str(a)+'.png', a_im)
                    #cv2.imwrite(b_set[3][-17:-1]+'--'+str(i+j+1)+'.'+str(b)+'.png', b_im)
                    cv2.imwrite(b_set.setname[-17:-1]+'--'+str(i+j+1)+'.'+str(b)+'.png', b_im)


g.render('g1')



