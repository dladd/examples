##################
#  Post Process  #
##################

import os,sys
import numpy as np
import subprocess
import pylab
from matplotlib import pyplot as plt
from subprocess import Popen, PIPE
import FluidExamples1DUtilities as Utilities1D

def Post(nodes,arteryLabels,outputDirs,pdfFile):
    # Set the reference values
    Ls = 1000.0              # Length   (m -> mm)
    Ts = 1000.0              # Time     (s -> ms)
    Ms = 1000.0              # Mass     (kg -> g)

    # Set the time parameters
    cycleTime = 790.0
    numberOfCycles = 5
    startTime = 0.0
    solveTime      = cycleTime*numberOfCycles
    timeIncrement = 0.2
    outputFrequency = 10.0
    
    #Choose type of plot(s)
    plotFlow = True
    plotPressure = False
    plotNonreflect = False

    # Set up numpy arrays
    numberOfTimesteps = int((solveTime/timeIncrement)/outputFrequency)
    timeResult = np.zeros((numberOfTimesteps))
    numberOfNodes = len(nodes)
    flowResult = np.zeros((numberOfNodes,numberOfTimesteps))
    pressureResult = np.zeros((numberOfNodes,numberOfTimesteps))

    # Read nodes
    inputNodeNumbers = []
    branchNodeNumbers = []
    coupledNodeNumbers = []
    nodeCoordinates = []
    branchNodeElements = []
    terminalArteryNames = []
    RCRParameters = []
    filename='input/Reymond2009_Nodes.csv'
#    filename='input/VH135/Nodes_VH135.csv'
    Utilities1D.CsvNodeReader2(filename,inputNodeNumbers,branchNodeNumbers,coupledNodeNumbers,
                               nodeCoordinates,branchNodeElements,terminalArteryNames,RCRParameters)
    numberOfInputNodes     = len(inputNodeNumbers)
    totalNumberOfNodes     = len(nodeCoordinates)
    numberOfBranches       = len(branchNodeNumbers)
    numberOfTerminalNodes  = len(coupledNodeNumbers)

    # Read elements
    elementNodes = []
    elementArteryNames = []
    elementNodes.append([0,0,0])
    filename='input/Reymond2009_Elements.csv'
#    filename='input/VH135/Elements_VH135.csv'
    Utilities1D.CsvElementReader2(filename,elementNodes,elementArteryNames)
    numberOfElements = len(elementNodes)-1

    # Set up reference types
    refs = ['Reymond2009'] # ,'Reymond2011'
#    refDir = './reference/VH135/'
    refDir = './reference/'
    refTypes = ['Model','Experimental']
    colours = ['r','c','b','g','k']

    # Node number for data extraction
    nodeLabels = []

    #Read in all node data
    print("Reading output file data (this can take a while...)")
    flowNodeData = np.zeros((len(outputDirs),totalNumberOfNodes,numberOfTimesteps))
    pressureNodeData = np.zeros((len(outputDirs),totalNumberOfNodes,numberOfTimesteps))

    #for timestep in range(0,int(solveTime/timeIncrement),outputFrequency):
    outputType = -1
    for outputDir in outputDirs:
        outputType += 1
        for timestep in range(numberOfTimesteps):
            filename = outputDir + "/MainTime_"+str(int(timestep*outputFrequency))+".part0.exnode"
            #print(filename)

            outputFile = open(filename)
            lines = outputFile.readlines()
            for i in range(0,len(lines)):
                if 'Node:' in lines[i]:
                    nodeNumber = [int(s) for s in lines[i].split() if s.isdigit()][0]
                    #print(nodeNumber)
                    # Extract the variables from output files
                    if nodeNumber == 9:
                        Flow = float(lines[i+4].split()[2])
                        Pressure = float(lines[i+12].split()[2])
                    else:
                        Flow = float(lines[i+4].split()[0])
                        Pressure = float(lines[i+12].split()[0])
                    Pressure = Pressure*Ls*(Ts**2)/(Ms*133.322)

                    # Store results
                    #print(nodeNumber)
                    flowNodeData[outputType,nodeNumber-1,timestep] = Flow
                    pressureNodeData[outputType,nodeNumber-1,timestep] = Pressure
            outputFile.close()
    timeResult =  np.arange(startTime,solveTime+startTime,timeIncrement*outputFrequency)
    #timeResult =  np.arange(0.0,solveTime/Ts,timeIncrement*outputFrequency/Ts)

    if plotFlow:
        nodeCounter   = 0
        for node in nodes:
            # Get artery names
            arteryLabel = arteryLabels[nodeCounter]
            print('Generating plot for ' + arteryLabel)
            nodeLabels.append(arteryLabel)

            # Do a subplot if looking at several nodes at once
            if numberOfNodes > 1:
                plt.subplot(int((numberOfNodes)/2)+numberOfNodes%2,2,nodeCounter+1)
            colourNum = 0
            # Plot Reymond data
            for ref in refs:
                for refType in refTypes:
                    filename = refDir +ref+'/'+refType+'/node_'+str(node)
                    if os.path.exists(filename):
                        refData = np.genfromtxt(filename,delimiter='\t')
                        numberOfRefTimesteps = len(refData[:,0])
                        refLabel = ref + ' ' + refType# + ' ' + arteryLabel
                        print('Found reference from '+refLabel)
                        refDataCycles=refData
                        addTime = np.zeros(numberOfRefTimesteps*numberOfCycles)
                        for cycle in range(1,int(numberOfCycles)):
                            refDataCycles = np.vstack((refDataCycles,refData))
                            addTime[numberOfRefTimesteps*cycle:]+=cycleTime
                        refDataCycles[:,0]=np.add(refDataCycles[:,0],addTime)
                        plt.plot(refDataCycles[:,0],refDataCycles[:,1],colours[colourNum],alpha=1.0,label=refLabel)
                        colourNum+=1
            colourNum = 0
            # Plot this node
            plt.title(arteryLabel+' (Node '+str(node)+')')
            if 'radial' in arteryLabel:
                plt.plot(timeResult,pressureNodeData[0,node-1,:],'k-',label='OpenCMISS Model')
                if plotNonreflect:
                    # Plot this node
                    plt.plot(timeResult,pressureNodeData[1,node-1,:],'k--',label='OpenCMISS Model Nonreflecting')
            else:
                plt.plot(timeResult,flowNodeData[0,node-1,:],'k-',label='OpenCMISS Model')
                if plotNonreflect:
                    # Plot this node
                    plt.plot(timeResult,flowNodeData[1,node-1,:],'k--',label='OpenCMISS Model Nonreflecting')

            if 'radial' in arteryLabel:
                plt.ylabel('Pressure (mmHg)')
            elif nodeCounter % 2 == 0:
                plt.ylabel(r'Flow rate (mLs$^{-1}$)')
            if nodeCounter > numberOfNodes - 3:
                plt.xlabel('Time (ms)')
            nodeCounter = nodeCounter+1        
            plt.xlim(startTime,solveTime+startTime)
        # Plot all nodes
        #leg = ax.legend(['abc'], loc = 'center left', bbox_to_anchor = (1.0, 0.5))
        plt.tight_layout()
        legend = [plt.legend(loc = (-0.6, -1.1))]
        if pdfFile != '':
            #plt.savefig(pdfFile, format='pdf')#, bbox_inches='tight')
            #plt.savefig(pdfFile,format='pdf', bbox_extra_artists=(plt.legend,), bbox_inches='tight')
            plt.savefig(pdfFile,format='pdf', bbox_extra_artists=legend, bbox_inches='tight')
        else:
            plt.show()
        #plt.savefig('temp.pdf', format='pdf',bbox_extra_artists=(plt.legend,), bbox_inches='tight')

    if plotPressure:
        nodeCounter   = 0
        for node in nodes:
            # Get artery names
            arteryLabel = arteryLabels[nodeCounter]
            print('Generating plot for ' + arteryLabel)
            nodeLabels.append(arteryLabel)

            # Do a subplot if looking at several nodes at once
            if numberOfNodes > 1:
                plt.subplot(int((numberOfNodes)/2)+numberOfNodes%2,2,nodeCounter+1)

            if nodeCounter % 2 == 0:
                plt.ylabel('Pressure (mmHg)')
            if nodeCounter > numberOfNodes - 3:
                plt.xlabel('Time (ms)')
            # Plot this node
            plt.plot(timeResult,pressureNodeData[0,node-1,:],'b-')
            plt.title(arteryLabel+' (Node '+str(node)+')')
            if plotNonreflect:
                # Plot this node
                plt.plot(timeResult,pressureNodeData[1,node-1,:],'k--',label='OpenCMISS Model Nonreflecting')
            nodeCounter = nodeCounter+1        
            plt.xlim(startTime,solveTime+startTime)
        # Plot all nodes
        plt.show()
    return

if len(sys.argv) > 1:
    nodes = int(sys.argv[1:])
else:
#    nodes = [1,9,16,82,100,30]#36,44,35]

    nodes = [100,102,104]#36,44,35]
    labels = ['femoral','anterior tibial','posterior tibial']
    outputDirs = ['output/Reymond2009ExpInputNonreflecting_new']#_pOutRCR100Hg',

#    nodes = [1,64,81,87,106,35]#36,44,35]
#    labels = ['Aortic root','Thoracic aorta','Abdominal aorta',
#              'Left common iliac','Right femoral','Right radial pressure']

#    nodes = [1,81,87,106]#36,44,35]
#    labels = ['Aortic root','Abdominal aorta','Left common iliac','Right femoral']
#    nodes = [1,10,16,82,100]#36,44,35]
#    labels = ['aortic root','thoracic aorta','abdominal aorta','left common iliac','right femoral']
              #,'Right common carotid',
#              'Left internal carotid','Right vertebral']
#    outputDirs = ['output/Reymond2009ExpInputCellML']#_pOutRCR100Hg',
                 # 'output/Reymond2009ExpInputNonreflecting']
#    outputDirs = ['output/Reymond2009ExpInputNonreflecting_pExt70']#_pOutRCR100Hg',
#    pdfFile = '/hpc/dlad004/thesis/Thesis/figures/cfd/55ArteryNonReflectingQP.pdf'

#    outputDirs = ['/media/F0F095D5F095A300/opencmissStorage/1D/Reymond/Reymond2009ExpInputCellML_40']

pdfFile = ''
Post(nodes,labels,outputDirs,pdfFile)

print "Processing Completed!"

#Popen(['gnuplot','PlotNode.p'])

