# ****************************************************************************
# * (C) Copyright 2020, Texas Instruments Incorporated. - www.ti.com
# ****************************************************************************
# *
# *  Redistribution and use in source and binary forms, with or without
# *  modification, are permitted provided that the following conditions are
# *  met:
# *
# *    Redistributions of source code must retain the above copyright notice,
# *    this list of conditions and the following disclaimer.
# *
# *    Redistributions in binary form must reproduce the above copyright
# *    notice, this list of conditions and the following disclaimer in the
# *     documentation and/or other materials provided with the distribution.
# *
# *    Neither the name of Texas Instruments Incorporated nor the names of its
# *    contributors may be used to endorse or promote products derived from
# *    this software without specific prior written permission.
# *
# *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
# *  PARTICULAR TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# *  A PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT  OWNER OR
# *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# *  EXEMPLARY, ORCONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# *  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# *  LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY, OR TORT (INCLUDING
# *  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *
# ****************************************************************************


# ****************************************************************************
# Sample mmW demo UART output parser script - should be invoked using python3
#       ex: python3 mmw_demo_example_script.py <recorded_dat_file_from_Visualizer>.dat
#
# Notes:
#   1. The parser_mmw_demo script will output the text version 
#      of the captured files on stdio. User can redirect that output to a log file, if desired
#   2. This example script also outputs the detected point cloud data in mmw_demo_output.csv 
#      to showcase how to use the output of parser_one_mmw_demo_output_packet
# ****************************************************************************


import serial
import time
import numpy as np
import os
import sys
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
# import the parser function 
from parser_mmw_demo import parser_one_mmw_demo_output_packet

from scipy import io
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

##DBSCAN을 위한 import
from plotnine import *
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# 시간측정을 위한 임시 변수
a = 1
b = 1
c = 1
d = 1
e = 1
f = 1
g = 1
h = 1
i = 1

frame = cv2.TickMeter()
f1 = 0

# Change the configuration file name
configFileName = 'xwr68xxconfig.cfg'

# Change the debug variable to use print()
DEBUG = False

# Constants
maxBufferSize = 2 ** 15;
CLIport = {}
Dataport = {}
byteBuffer = np.zeros(2 ** 15, dtype='uint8')
byteBufferLength = 0;
maxBufferSize = 2 ** 15;
magicWord = [2, 1, 4, 3, 6, 5, 8, 7]
detObj = {}
frameData = {}
currentIndex = 0
# word array to convert 4 bytes to a 32 bit number
word = [1, 2 ** 8, 2 ** 16, 2 ** 24]


# Function to configure the serial ports and send the data from
# the configuration file to the radar
def serialConfig(configFileName):
    global a
    tm = cv2.TickMeter()
    tm.reset()
    tm.start()
    t1 = time.time()

    global CLIport
    global Dataport
    # Open the serial ports for the configuration and the data ports

    # Raspberry pi
    CLIport = serial.Serial('/dev/ttyUSB0', 115200)
    Dataport = serial.Serial('/dev/ttyUSB1', 921600)

    # Windows
    #CLIport = serial.Serial('COM14', 115200)
    #Dataport = serial.Serial('COM15', 921600)

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        CLIport.write((i + '\n').encode())
        print(i)
        time.sleep(0.01)
    tm.stop()
    ms = tm.getTimeMilli()
    print('serialConfig time:', (time.time() - t1) * 1000, 'ms')
    print(a, "번째 프레임")
    a = a + 1
    return CLIport, Dataport


# Function to parse the data inside the configuration file
def parseConfigFile(configFileName):
    global b
    tm = cv2.TickMeter()
    tm.reset()
    tm.start()
    t1 = time.time()

    configParameters = {}  # Initialize an empty dictionary to store the configuration parameters

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:

        # Split the line
        splitWords = i.split(" ")

        # Hard code the number of antennas, change if other configuration is used
        numRxAnt = 4
        numTxAnt = 3

        # Get the information about the profile configuration
        if "profileCfg" in splitWords[0]:
            startFreq = int(float(splitWords[2]))
            idleTime = int(splitWords[3])
            rampEndTime = float(splitWords[5])
            freqSlopeConst = float(splitWords[8])
            numAdcSamples = int(splitWords[10])
            numAdcSamplesRoundTo2 = 1;

            while numAdcSamples > numAdcSamplesRoundTo2:
                numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2;

            digOutSampleRate = int(splitWords[11]);

        # Get the information about the frame configuration
        elif "frameCfg" in splitWords[0]:

            chirpStartIdx = int(splitWords[1]);
            chirpEndIdx = int(splitWords[2]);
            numLoops = int(splitWords[3]);
            numFrames = int(splitWords[4]);
            framePeriodicity = int(splitWords[5]);

    # Combine the read data to obtain the configuration parameters
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame / numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (
                2 * freqSlopeConst * 1e12 * numAdcSamples)
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (
                2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / (
                2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate) / (2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)

    tm.stop()
    ms = tm.getTimeMilli()
    print('parseConfigFile time:', (time.time() - t1) * 1000, 'ms')
    print(b, "번째 프레임")
    b = b + 1

    return configParameters


##################################################################################
# USE parser_mmw_demo SCRIPT TO PARSE ABOVE INPUT FILES
##################################################################################
def readAndParseData14xx(Dataport, configParameters):
    global c
    tm = cv2.TickMeter()
    tm.reset()
    tm.start()
    t1 = time.time()
    # load from serial
    global byteBuffer, byteBufferLength

    # Initialize variables
    magicOK = 0  # Checks if magic number has been read
    dataOK = 0  # Checks if the data has been read correctly
    frameNumber = 0
    detObj = {}

    readBuffer = Dataport.read(Dataport.in_waiting)
    byteVec = np.frombuffer(readBuffer, dtype='uint8')
    byteCount = len(byteVec)

    # Check that the buffer is not full, and then add the data to the buffer
    if (byteBufferLength + byteCount) < maxBufferSize:
        byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]
        byteBufferLength = byteBufferLength + byteCount

    # Check that the buffer has some data
    if byteBufferLength > 16:

        # Check for all possible locations of the magic word
        possibleLocs = np.where(byteBuffer == magicWord[0])[0]

        # Confirm that is the beginning of the magic word and store the index in startIdx
        startIdx = []
        for loc in possibleLocs:
            check = byteBuffer[loc:loc + 8]
            if np.all(check == magicWord):
                startIdx.append(loc)

        # Check that startIdx is not empty
        if startIdx:

            # Remove the data before the first start index
            if startIdx[0] > 0 and startIdx[0] < byteBufferLength:
                byteBuffer[:byteBufferLength - startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
                byteBuffer[byteBufferLength - startIdx[0]:] = np.zeros(len(byteBuffer[byteBufferLength - startIdx[0]:]),
                                                                       dtype='uint8')
                byteBufferLength = byteBufferLength - startIdx[0]

            # Check that there have no errors with the byte buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0

            # Read the total packet length
            totalPacketLen = np.matmul(byteBuffer[12:12 + 4], word)
            # Check that all the packet has been read
            if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                magicOK = 1

    # If magicOK is equal to 1 then process the message
    if magicOK:
        # Read the entire buffer
        readNumBytes = byteBufferLength
        if (DEBUG):
            print("readNumBytes: ", readNumBytes)
        allBinData = byteBuffer
        if (DEBUG):
            print("allBinData: ", allBinData[0], allBinData[1], allBinData[2], allBinData[3])

        # init local variables
        totalBytesParsed = 0;
        numFramesParsed = 0;

        # parser_one_mmw_demo_output_packet extracts only one complete frame at a time
        # so call this in a loop till end of file
        #
        # parser_one_mmw_demo_output_packet function already prints the
        # parsed data to stdio. So showcasing only saving the data to arrays
        # here for further custom processing
        parser_result, \
        headerStartIndex, \
        totalPacketNumBytes, \
        numDetObj, \
        numTlv, \
        subFrameNumber, \
        detectedX_array, \
        detectedY_array, \
        detectedZ_array, \
        detectedV_array, \
        detectedRange_array, \
        detectedAzimuth_array, \
        detectedElevation_array, \
        detectedSNR_array, \
        detectedNoise_array = parser_one_mmw_demo_output_packet(allBinData[totalBytesParsed::1],
                                                                readNumBytes - totalBytesParsed, DEBUG)

        # Check the parser result
        if (DEBUG):
            print("Parser result: ", parser_result)
        if (parser_result == 0):
            totalBytesParsed += (headerStartIndex + totalPacketNumBytes)
            numFramesParsed += 1
            if (DEBUG):
                print("totalBytesParsed: ", totalBytesParsed)
            ##################################################################################
            # TODO: use the arrays returned by above parser as needed.
            # For array dimensions, see help(parser_one_mmw_demo_output_packet)
            # help(parser_one_mmw_demo_output_packet)
            ##################################################################################

            # For example, dump all S/W objects to a csv file
            """
            import csv
            if (numFramesParsed == 1):
                democsvfile = open('mmw_demo_output.csv', 'w', newline='')                
                demoOutputWriter = csv.writer(democsvfile, delimiter=',',
                                        quotechar='', quoting=csv.QUOTE_NONE)                                    
                demoOutputWriter.writerow(["frame","DetObj#","x","y","z","v","snr","noise"])            

            for obj in range(numDetObj):
                demoOutputWriter.writerow([numFramesParsed-1, obj, detectedX_array[obj],\
                                            detectedY_array[obj],\
                                            detectedZ_array[obj],\
                                            detectedV_array[obj],\
                                            detectedSNR_array[obj],\
                                            detectedNoise_array[obj]])
            """
            detObj = {"numObj": numDetObj, "range": detectedRange_array, \
                      "x": detectedX_array, "y": detectedY_array, "z": detectedZ_array}
            dataOK = 1
        else:
            # error in parsing; exit the loop
            print("error in parsing this frame; continue")

        shiftSize = totalPacketNumBytes
        byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength]
        byteBuffer[byteBufferLength - shiftSize:] = np.zeros(len(byteBuffer[byteBufferLength - shiftSize:]),
                                                             dtype='uint8')
        byteBufferLength = byteBufferLength - shiftSize

        # Check that there are no errors with the buffer length
        if byteBufferLength < 0:
            byteBufferLength = 0
        # All processing done; Exit
        if (DEBUG):
            print("numFramesParsed: ", numFramesParsed)

    tm.stop()
    ms = tm.getTimeMilli()
    print('readAndParseData14xx time:', (time.time() - t1) * 1000, 'ms')
    print(c, "번째 프레임")
    c = c + 1

    return dataOK, frameNumber, detObj


class MyWidget(pg.GraphicsLayoutWidget):

    def __init__(self, parent=None):
        global d
        tm = cv2.TickMeter()
        tm.reset()
        tm.start()
        t1 = time.time()
        super().__init__(parent=parent)
        #print("self : ",self)
        self.mainLayout = QtWidgets.QVBoxLayout()
        self.setLayout(self.mainLayout)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(100)  # in milliseconds
        self.timer.start()
        self.timer.timeout.connect(self.onNewData)
        #print("self.onNewData",self.onNewData)
        self.plotItem = self.addPlot(title="Radar points")
        #print("self.plotItem",self.plotItem)



        self.plotDataItem = self.plotItem.plot([], pen=None,
                                               symbolBrush=(255, 0, 0), symbolSize=2, symbolPen='r')#symbolsize를 통해 점의 좌표크기를 나타냄, 작을수록 더 많은 좌표 출력가능



        tm.stop()
        ms = tm.getTimeMilli()
        print('MyWidget(pg.GraphicsLayoutWidget): time:', (time.time() - t1) * 1000, 'ms')
        print(d, "번째 프레임")
        d = d + 1

    def setData(self, x, y):
        #print('sns버전@@@', sns.__version__)

        global e
        tm = cv2.TickMeter()
        tm.reset()
        tm.start()
        t1 = time.time()

        print("X : ", x)
        print("Y : ", y)
        ###
        tm.stop()
        ms = tm.getTimeMilli()
        #print('setData time:', (time.time() - t1) * 1000, 'ms')
        print(e, "번째 프레임")
        e = e + 1
        self.plotDataItem.setData(x, y)

    # Funtion to update the data and display in the plot
    def update(self):
        global f
        global frame
        global f1
        frame.stop()
        ms = frame.getTimeMilli()
        print('Frame time:', (time.time() - f1) * 1000, 'ms')
        print(f, "번째 프레임")

        # 프레임당 걸리는 시간 측정
        frame.reset()
        frame.start()
        f1 = time.time()

        # update 함수 걸리는 시간 측정
        tm = cv2.TickMeter()
        tm.reset()
        tm.start()
        t1 = time.time()

        # printf가 걸리는 시간 측정
        printTime = cv2.TickMeter()
        printTime.reset()
        printTime.start()
        P1 = time.time()
        print('print문이 걸리는 시간 : ', (time.time() - P1) * 1000, 'ms')

        dataOk = 0
        global detObj
        x = []
        y = []

        # Read and parse the received data
        dataOk, frameNumber, detObj = readAndParseData14xx(Dataport, configParameters)
        if dataOk and len(detObj["x"]) > 0:
            # print(detObj)
            x = detObj["x"]
            y = detObj["y"]
            print("X좌표 : ",x)
            print("Y좌표 : ",y)

            # DBSCAN 적용
            names_dic = {'x': x, 'y': y}
            d1 = pd.DataFrame(names_dic)
            z = StandardScaler()
            d1[["x", "y"]] = z.fit_transform(d1[["x", "y"]])
            #print("d1[[""x"", ""y""]]",d1[["x", "y"]])
            db1 = DBSCAN(eps=0.3, min_samples=3).fit(d1)
            labsList = ["Noise"]
            labsList = labsList + ["Cluster" + str(i) for i in range(1, len(set(db1.labels_)))]

            #객체의 좌표 출력
            classid = pd.DataFrame(db1.labels_, columns=['id'])

            #데이터 프레임 병합
            final = d1.join(classid)
            print("객체로 분류된 좌표")
            print(final)
            #각 객체별 평균 좌표
            grouped = final.groupby(final['id']).mean()

            print("len(grouped) : ",len(grouped))
            #모든 객체 평균 좌표 출력


            if -1 in db1.labels_:
                for j in range(0, len(grouped)-1):
                    print(j, "번째 객체 평균 좌표 : ")
                    print(grouped.loc[j])
            else:
                for j in range(0, len(grouped)):
                    print(j, "번째 객체 평균 좌표 : ")
                    print(grouped.loc[j])




           # print("db1.labels_ : ",db1.labels_)
            #print("set(db1.labels_) : ",set(db1.labels_))
            print("DBSCAN 결과 : ",labsList)
            d1["assignments"] = db1.labels_

            #print("d1[""assignments""] : " ,d1["assignments"])

            #print("d1[""assignments""][x] : ", d1["assignments"]["x"])
            #print("d1[""assignments""][y] : ", d1["assignments"]["y"])
            (ggplot(d1,
                    aes(x="x", y="y",
                        color="factor(assignments)")) + geom_point() + theme_minimal() + scale_color_discrete(
                name="Cluster Assignment", labels=labsList) + theme(panel_grid_major=element_blank()) + labs(
                title="DBSCAN with eps=0.5, min_sample=5"))





        tm.stop()
        ms = tm.getTimeMilli()
        #print('update time:', (time.time() - t1) * 1000, 'ms')
        print(f, "번째 프레임")
        f = f + 1
        return dataOk, x, y

    def onNewData(self):
        global g
        tm = cv2.TickMeter()
        tm.reset()
        tm.start()
        t1 = time.time()
        #####

        # Update the data and check if the data is okay
        dataOk, newx, newy = self.update()

        # if dataOk:
        # Store the current frame into frameData
        # frameData[currentIndex] = detObj
        # currentIndex += 1

        x = newx
        y = newy
        #####
        tm.stop()
        ms = tm.getTimeMilli()
        #print('onNewData time:', (time.time() - t1) * 1000, 'ms')
        print(g, "번째 프레임")
        g = g + 1
        self.setData(x, y)


def main():
    h = 1
    # Configurate the serial port
    CLIport, Dataport = serialConfig(configFileName)
    # Get the configuration parameters from the configuration file
    global configParameters
    configParameters = parseConfigFile(configFileName)

    # dbscan = DBSCAN(x, 0.3, 4)
    # 이웃과의 거리를 나타내는 최소 이웃 반경 ϵ 과 최소 이웃 수 minPts입니다.

    app = QtWidgets.QApplication([])
    tm = cv2.TickMeter()
    tm.reset()
    tm.start()
    t1 = time.time()

    pg.setConfigOptions(antialias=False)  # True seems to work as well
    tm.stop()
    ms = tm.getTimeMilli()
    print(' pg.setConfigOptions(antialias=False) time:', (time.time() - t1) * 1000, 'ms')
    print(h, "번째 프레임")
    h = h + 1
    win = MyWidget()
    win.show()
    win.resize(800, 600)
    win.raise_()
    app.exec_()
    CLIport.write(('sensorStop\n').encode())
    CLIport.close()
    Dataport.close()


if __name__ == "__main__":
    main()