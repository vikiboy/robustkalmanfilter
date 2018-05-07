import cv2 as cv
from math import cos, sin, sqrt
import numpy as np
import matplotlib.pyplot as plt


def regressionMode(state):
    """
    batch mode regression mode
    -- equation 13
    """
    hMat = np.concatenate((kalman.measurementMatrix,np.eye(2)),axis=0)

    zeroMat = np.zeros((2,2))
    zeroMat[0,0] = kalman.measurementNoiseCov
    zeroMat[0,1] = kalman.measurementNoiseCov
    zeroMat[1,0] = kalman.measurementNoiseCov
    zeroMat[1,1] = kalman.measurementNoiseCov

    rCov = np.hstack((np.vstack((zeroMat,np.zeros((2,2)))),np.vstack((np.zeros((2,2)),kalman.errorCovPre))))
    rCov1 = [[zeroMat[0,0],0,0],[0,kalman.errorCovPre[0,0],0],[0,0,kalman.errorCovPre[1,1]]]

    l = np.linalg.cholesky(rCov1)

    mError = kalman.measurementNoiseCov * np.random.randn(1, 1) # measurement error
    pError = sqrt(kalman.errorCovPre[0,0]) * np.random.randn(2, 1) # process error

    batchOutput = np.dot(hMat,state) + np.vstack((mError,pError))

    return batchOutput,l


def psOutliers(h):
    """
    Projection Statistics
    """

    u0 = h[:,0]-np.median(h[:,0])
    u1 = h[:,1]-np.median(h[:,1])
    u2 = h[:,2]-np.median(h[:,2])

    u = []
    v = []

    for i in range(len(h)):
        u.append([u0[i],u1[i],u2[i]]) # each data point
        if(i>0):
            tempV = 1/sqrt(u0[i]**2 + u1[i]**2 + u2[i]**2)
            v = np.vstack((v,[u0[i]*tempV,u1[i]*tempV,u2[i]*tempV]))
        else:
            v = np.array([u0[i],u1[i],u2[i]])

    z = np.dot(h,v.T).T


    if(z.shape[0]==1):
        zMed = np.array([0.0])
    else:
        zMed = np.median(z,axis=1)

    c = 1+15.0/(len(h)-3)

    zMed = zMed.reshape(len(zMed),1)

    MAD = []

    for i in range(len(h)):
        mad = []
        for j in range(len(h)):
            mad.append(abs(z[i,j]-zMed[i]))
        medMad = np.median(mad)
        MAD.append(1.4826*c*medMad)

    PS = np.zeros((len(h),len(h)))

    for i in range(len(h)):
        for j in range(len(h)):
            PS[i,j] = abs(z[i,j]-zMed[i])/MAD[i]

    psMax = np.max(PS,axis=1)
    return psMax[-1]

def prewhitening(h,L,state):
    """
    Perform the outlier detection
    PS --> weights ---> (15) ---> (17)
    """
    hPre = h
    resPre = h

    d =  1.5
    varPi = min(1,2.25/psOutliers(h)**2)
    h[-1] = h[-1]*varPi
    hPre[-1] = np.dot(np.linalg.inv(L),h[-1])

    hMat = np.concatenate((kalman.measurementMatrix,np.eye(2)),axis=0)
    hMatinv = np.dot(np.linalg.inv(L),hMat)
    stateInv = np.dot(hMatinv,state)
    resPre[-1] = hPre[-1] - stateInv.T
    s = [np.median(resPre[:,0]),np.median(resPre[:,1]),np.median(resPre[:,2])]
    rSi = [resPre[-1][0]/(varPi*np.median(resPre[:,0])),resPre[-1][1]/(varPi*np.median(resPre[:,1])),resPre[-1][2]/(varPi*np.median(resPre[:,2]))]
    rSi_mod = sqrt(rSi[0]**2 + rSi[1]**2 + rSi[2]**2)
    drho_rSi = [1.5,1.5,1.5]

    if(rSi_mod<-1.5):
        drho_rSi = [-1.5,-1.5,-1.5]
    elif(rSi_mod>1.5):
        drho_rSi = [1.5,1.5,1.5]
    else:
        drho_rSi = rSi

    q_rSi = [drho_rSi[0]/rSi[0],drho_rSi[1]/rSi[1],drho_rSi[2]/rSi[2]]
    Q_mat = [[q_rSi[0],0,0],[0,q_rSi[1],0],[0,0,q_rSi[2]]]
    return Q_mat, hMat, hPre, varPi


def finalEstimation(h,L,state):
    """
    Solve final regression and update covariance matrix
    (22)
    """
    Q_mat, hMat, hPre, varPi = prewhitening(h,L,state)
    hCap = np.linalg.inv(np.dot(np.dot(hMat.T,Q_mat),hMat))
    finalState = np.dot(np.dot(np.dot(hCap,hMat.T),Q_mat),hPre[-1])

    Q_vpi = [[varPi,0,0],[0,varPi,0],[0,0,varPi]]

    finalCov = np.linalg.inv(np.dot(hMat.T,hMat))
    finalCov = np.dot(finalCov,np.dot(np.dot(hMat.T,Q_vpi),hMat))
    finalCov = np.dot(finalCov,np.linalg.inv(np.dot(hMat.T,hMat)))
    finalCov = finalCov * 1.04

    return finalState,finalCov

if __name__ == '__main__':
    img_height = 500
    img_width = 500
    kalman = cv.KalmanFilter(2,1,0)
    while True:
        state = 0.1 * np.random.randn(2, 1)

        kalman.transitionMatrix = np.array([[1., 1.], [0., 1.]]) #A
        kalman.measurementMatrix = 1. * np.ones((1, 2)) #H
        kalman.processNoiseCov = 1e-5 * np.eye(2) #Q
        kalman.measurementNoiseCov = 1e-1 * np.ones((1, 1)) #R
        kalman.errorCovPost = 1. * np.ones((2, 2)) #updated posteriori error estimate covariance matrix
        kalman.statePost = 0.1 * np.random.randn(2, 1) #corrected state

        count = 0

        X_state = []
        Y_state = []
        X_finalState = []
        Y_finalState = []

        while True:
            def calc_point(angle):
                return (np.around(img_width/2 + img_width/3*cos(angle), 0).astype(int),
                        np.around(img_height/2 - img_width/3*sin(angle), 1).astype(int))

            # plot points
            def draw_cross(center, color, d):
                cv.line(img,
                         (center[0] - d, center[1] - d), (center[0] + d, center[1] + d),
                         color, 1, cv.LINE_AA, 0)
                cv.line(img,
                         (center[0] + d, center[1] - d), (center[0] - d, center[1] + d),
                         color, 1, cv.LINE_AA, 0)

            state_angle = state[0, 0]
            state_pt = calc_point(state_angle)
            prediction = kalman.predict() #prediction, #errorCovPre

            predict_angle = prediction[0, 0]
            predict_pt = calc_point(predict_angle)

            img = np.zeros((img_height, img_width, 3), np.uint8)
            draw_cross(np.int32(state_pt), (255, 255, 255), 3)
            draw_cross(np.int32(predict_pt), (0, 255, 0), 3)
            cv.line(img, state_pt, predict_pt, (0, 255, 255), 3, cv.LINE_AA, 0)

            batchProcess,L = regressionMode(state)
            if(count==0):
                hPS = batchProcess.T
            else:
                hPS = np.vstack((hPS,batchProcess.T))
            count+=1

            if(count>3):
                finalState,finalCov = finalEstimation(hPS,L,state)
                process_noise = sqrt(kalman.processNoiseCov[0,0]) * np.random.randn(2, 1)
                state = np.dot(kalman.transitionMatrix, state) + process_noise

                # if(count<50):
                #     state = state*10

                kalman.statePost = finalState
                kalman.errorCovPost = finalCov

                X_state.append(state[0])
                Y_state.append(state[1])

                X_finalState.append(prediction[0])
                Y_finalState.append(prediction[1])

                if(count in [100,200,300,400,500]):
                    plt.scatter(X_state,Y_state,color='red')
                    plt.scatter(X_finalState,Y_finalState,color='green')
                    plt.show()
                    count = 0

                cv.imshow("Kalman", img)

                code = cv.waitKey(100)
                if code != -1:
                    break
        if code in [27, ord('q'), ord('Q')]:
            break

    cv.destroyWindow("Kalman")
