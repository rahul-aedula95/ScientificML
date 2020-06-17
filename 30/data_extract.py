###########################################################################################################################################################
# This code is directly ported from Shreyas Honrao's personal repository upon request. The complete rights for this code belong to Shreyas Honrao et.al. 
###########################################################################################################################################################





import numpy as np
import scipy
import pymatgen as pmg
import os.path
import pickle
from pymatgen.io.vasp import Poscar
import itertools




def parseData(directoryPath, elements = ["Li", "Ge"], pairTup = [["Li","Li"], ["Li","Ge"], ["Ge","Ge"]]):
    """
    Creates an input data array for the ML algorithm. For each structure - its RDFMatrix, formation energy, 
    molar fraction, groupnumber and stepnumber are stored.

    Args:
        directoryPath: path to Archive data folder.

        elements: list of elements in the binary system, in this case elem_A and elem_B.

        pairTup: list of all element pairs for which the partial RDF is calculated.

    """

    fileList = os.listdir(directoryPath)
    print("Total number of structures:", len(fileList)/2)
    fileList = sorted(fileList) #sorts files so that the energy file is before the poscar file for every structure

    numElements = len(elements)

    rawData = []
    for i in range(0,len(fileList),2):
        splitStr = fileList[i].split(".")
        dftStr = splitStr[0].split("_")
        dftGroupNum = int(dftStr[1]) #every GA relaxation counts as a different dftGroup
        stepNum = int(dftStr[2])     #every ionic step within the relaxation is called a step

        energyFilePath = os.path.join(directoryPath, fileList[i])
        energyFileEntry = open(energyFilePath, 'r')
        energy = float(energyFileEntry.readline()) #reads in **energy per atom** for the structure from Archive

        poscarFilePath = os.path.join(directoryPath, fileList[i+1])
        tempStruct = Poscar.from_file(poscarFilePath, False).structure
        RDFMatrix = getRDF_Mat(tempStruct, pairTup) #gets RDFMatrix of the structure (for all element pairs listed in the pairTup) using the getRDF_Mat function defined below
        
        molarFracs = [] #calculates the molar fraction of elem_A and elem_B in the structure
        for i in range(0, numElements):
            elem = pmg.Element(elements[i])
            elemPerc = tempStruct.composition.get_atomic_fraction(elem)
            molarFracs.append((elements[i], elemPerc))

        molarFracs = dict(molarFracs) 
        
        rawData.append([RDFMatrix, energy, molarFracs, dftGroupNum, stepNum]) #creates input data array
   
    refEnergies = [] #gets the reference energies for elem_A and elem_B, calculated as the lowest energy of pure elem crystals in the dataset
    for elem in elements:
        pureElemCrystals = [crystal for crystal in rawData if crystal[2][elem] == 1.0]
        energies = list(zip(*pureElemCrystals))[1]
        minEnergy = min(energies)
        refEnergies.append((elem, minEnergy))
    refEnergies = dict(refEnergies)
     
    formTransData = [] #calculates formation energies of all structures from reference energies of elem_A and elem_B, replaces total energies in input data array with formation eenrgies 
    for datum in rawData:
        molarFracs = datum[2]
        formEnergy = datum[1]
    
        if numElements > 1:
            for elem in elements:
                formEnergy = formEnergy - molarFracs[elem]*refEnergies[elem]

        else:
            elem = elements[0]
            formEnergy = formEnergy - refEnergies[elem]

        datum[1] = formEnergy
        formTransData.append(datum)

    groupNumbers = set(map(lambda x:x[3], formTransData))  #groups all ionic steps from a single GA run together 
    groupedData = [[crystal for crystal in formTransData if crystal[3] == groupNum] for groupNum in groupNumbers]
    #pickle.dump(groupedData, open("LiGe_RDF.p","wb"))
    return(groupedData) 



def getRDF_Mat(cell, pairTup, cutOffRad = 10.0, sigma = 0.2, stepSize = 0.1):
    """
    Calculates the RDF for every structure.

    Args:
        cell: input structure.

        pairTup: list of all element pairs for which the partial RDF is calculated.

        cutOffRad: max. distance up to which atom-atom intereactions are considered.

        sigma: width of the Gaussian, used for broadening

        stepSize:  bin width, binning transforms the RDF into a discrete representation. 

    """

    binRad = np.arange(0, cutOffRad, stepSize) #makes bins based on stepSize and cutOffRad
    numBins = len(binRad)
    numPairs = len(pairTup)
    vec = np.zeros((numPairs, numBins)) #creates zero vector of size 3x100
    
    for index,pair in enumerate(pairTup):
        alphaSpec = pmg.Element(pair[0]) #alphaSpec and betaSpec are the two elements of a pair from the pairTup 
        betaSpec = pmg.Element(pair[1])
        hist = np.zeros(numBins)  
        neighbors = cell.get_all_neighbors(cutOffRad) #gets all neighboring atoms within cutOffRad for alphaSpec and betaSpec
    
        sites = cell.sites #all sites within unit cell
        #volume = pmg.core.structure.IStructure.from_sites(sites).volume #gets volme of unit cell
        indicesA = [j[0] for j in enumerate(sites) if j[1].specie == alphaSpec] #gets all alphaSpec sites within the unit cell
        numAlphaSites = len(indicesA)
        indicesB = [j[0] for j in enumerate(sites) if j[1].specie == betaSpec]  #gets all betaSpec sites within the unit cell
        numBetaSites = len(indicesB)
    
        if numAlphaSites == 0 or numBetaSites == 0:
            vec[index] = hist #partial RDF for an alphaSpec-betaSpec pair is zero everywhere if no alphaSpec or no betaSpec atoms are present in the structure
            continue    

        alphaNeighbors = [neighbors[i] for i in indicesA] #neighbors of alphaSpec only 
    
        alphaNeighborDistList = []
        for aN in alphaNeighbors:
            tempNeighborList = [neighbor for neighbor in aN if neighbor[0].specie==betaSpec] #neighbors of alphaSpec that are betaSpec
            alphaNeighborDist = []
            for j in enumerate(tempNeighborList):
                alphaNeighborDist.append(j[1][1])
            alphaNeighborDistList.append(alphaNeighborDist) #add the neighbor distances of all such neighbors to a list

        for aND in alphaNeighborDistList:
            for dist in aND: #apply gaussian broadening to the neigbor distances, so the effect of having a neighbor at distance x is spread out over few bins around x
                inds = dist/stepSize
                inds = int(inds)
                lowerInd = inds-5
                if lowerInd < 0:
                    while lowerInd < 0:
                        lowerInd = lowerInd + 1
                upperInd = inds+5
                if upperInd >= numBins:
                    while upperInd >= numBins:
                        upperInd = upperInd - 1
                ind = range(lowerInd, upperInd)
                evalRad = binRad[ind] 
                exp_Arg = .5 *( ( np.subtract( evalRad, dist)/ (sigma) )**2) #calculate RDF value for each bin
                rad2 = np.multiply(evalRad, evalRad) #add a 1/r^2 normalization term, check paper for descripton
                hist[ind] += np.divide(np.exp(-exp_Arg), rad2)
    
        tempHist = hist/numAlphaSites #divide by number of AlphaSpec atoms in the unit cell to give the final partial RDF
        vec[index] = tempHist
    
    vec = np.row_stack((vec[0], vec[1], vec[2]))  #combine all 3 partials RDFs to get RDFMatrix
    return vec


def filterEnergies(dftRuns, lowCutoff, highCutoff):
    
    filteredData = []
    mlData = list(itertools.chain(*dftRuns))
    
    for i in range(len(mlData)):
        if mlData[i][1] > lowCutoff and mlData[i][1] < highCutoff and mlData[i][2]:
            filteredData.append(mlData[i])
            
    groupNumIndice = 3
    groupNumbers = set(map(lambda x:x[groupNumIndice], filteredData))
    dftRuns = [[crystal for crystal in filteredData if crystal[groupNumIndice]==groupNum] for groupNum in groupNumbers]

    return(dftRuns)

if __name__ == "__main__":
    
    groupedData = parseData(directoryPath = "data/Archive/")
    filteredData = filterEnergies(groupedData, lowCutoff=-0.5, highCutoff=0.2)    
    
    with open('data/cleanedData.pickle', 'wb') as handle:
        pickle.dump(filteredData, handle, protocol=pickle.HIGHEST_PROTOCOL)