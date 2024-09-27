import time
import pandas as pd
import base64

from collections import Counter
from os import listdir
from os.path import isfile, join
from openai import OpenAI


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def prepareImgs():
    fileDict = {}
    onlyfiles = [f for f in listdir("IAA_CAP/") if isfile(join("IAA_CAP/", f))]
    for file in onlyfiles:
        imgId = file.split("_")[1]
        fileDict[imgId] = encode_image("IAA_CAP/{}".format(file))
    return fileDict

def completeListTrain():
    responseDictTrain = {}
    test = pd.read_csv("fixations/fixations_train.csv",sep="\t")
    for id,entry,response,participant in zip(list(test["ImageID"]),list(test["AOI Label"]),list(test["Answers"]),list(test["Respondent Name"])):
        if not participant in responseDictTrain.keys():
            responseDictTrain[participant] = [[entry,id,response]]
        else:
            responseDictTrain[participant].append([entry,id,response])
    return responseDictTrain

def completeListTest():
    responseDictTest = {}
    test = pd.read_csv("fixations/fixations_test.csv",sep="\t")
    for id,entry,response,participant in zip(list(test["ImageID"]),list(test["AOI Label"]),list(test["Answers"]),list(test["Respondent Name"])):
        if not participant in responseDictTest.keys():
            responseDictTest[participant] = [[entry,id,response]]
        else:
            responseDictTest[participant].append([entry,id,response])
    return responseDictTest

def responseTotxt(responseNr):
    if responseNr == 0:
        return "YES"
    elif responseNr == 1:
        return "NO"
    elif responseNr == 2:
        return "UNCLEAR"

def shotChoice(shots):
    shotsOut = []
    containsYes = False
    containsNo = False
    containsUnclear = False
    shotsPos = None
    shotsNeg = None
    shotsUnclear = None
    for shot in shots:
        if shot[-1] == 0 and not containsYes:
            shotsPos = shot
            containsYes = True
        if shot[-1] == 1 and not containsNo:
            shotsNeg = shot
            containsNo = True
        if shot[-1] == 2 and not containsUnclear:
            shotsUnclear = shot
            containsUnclear = True
    if shotsNeg == None and shotsUnclear == None:
        shotsOut.append(shots[0])
        shotsOut.append(shots[1])
    elif shotsNeg == None and not shotsUnclear == None:
        shotsOut.append(shotsPos)
        shotsOut.append(shotsUnclear)
    elif shotsUnclear == None and not shotsNeg == None:
        shotsOut.append(shotsPos)
        shotsOut.append(shotsNeg)
    elif not shotsUnclear == None and not shotsNeg == None:
        shotsOut.append(shotsPos)
        shotsOut.append(shotsNeg)
    return shotsOut

def generatePrompts(trainDict,testDict,fileDict,client):
    with open("gptLog.txt","a") as f:
                f.write("Participant\tImageID\tExpected\tPredicted\n")
    for participant in testDict.keys():
        fixations = testDict[participant]
        shots = trainDict[participant]
        shots = shotChoice(shots)
        for fixation in fixations:
            img1 = shots[0][-2] #one shot sample img
            img3 = shots[1][-2] #two shot sample img
            img2 = fixation[-2]
            #this is the two-shot setup. all setups are contained in the Prompts[...] txt files.
            messagesIn=[
                {"role": "system","content": [
                            {"type" : "text", "text" : "You are a study participant that is tasked to evaluate the consistency of an image and a corresponding caption. Your perception process is described in a list. This list indicates the sequence in which the stimulus elements were perceived. In this list, elements starting with 'txt_' denote elements of the caption and elements starting with 'vis_' denote elements of the image. 'off' denotes an unspecified image region. The question you have to answer is: 'Does the caption mention the central entities in the image?' There are three possible answers to the question: Yes, No, Unclear"},
                             {"type":"text","text":"Consider this first stimulus as a training example. The example stimulus and your perception sequence look like this:"},
                             {"type":"image_url","image_url": {"url":"data:image/jpeg;base64,{}".format(fileDict[str(img1)])}},{"type":"text","text":"Perception sequence: {}".format(shots[0][0])},
                             {"type": "text", "text":"The expected response based on this input is: {}".format(responseTotxt(shots[0][-1]))},
                             {"type":"text","text":"Consider this second stimulus as another training example. The example stimulus and your perception sequence look like this:"},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,{}".format(fileDict[str(img3)])}},
                             {"type":"text","text":"Perception sequence: {}".format(shots[1][0])},{"type": "text", "text":"The expected response based on this input is: {}".format(responseTotxt(shots[1][-1]))}]},
                {"role" : "user", "content":[
                            {"type":"text","text":"Here, the stimulus and the perception sequence are provided. Based on these inputs and your observed patterns from the training examples, answer the question and respond only with one of the three options"},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,{}".format(fileDict[str(img2)])}},{"type":"text","text":"Perception Sequence: '{}'".format(fixation[0])}]}
                ]
            response = client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=messagesIn,
                    max_tokens=100,
                )
            time.sleep(60) #to prevent errors related to token limits
            with open("gptLog.txt","a") as f:
                f.write("{}\t{}\t{}\t{}\n".format(participant,img2,responseTotxt(fixation[-1]),response.choices[0].message.content))


if __name__ == "__main__":
    responseTest = completeListTest()
    responseTrain = completeListTrain()
    imgDict = prepareImgs()

    key=""
    with open("oaKey","r") as keyfile:
        for line in keyfile:
            key += line

    client = OpenAI(api_key=key)
    generatePrompts(responseTrain,responseTest,imgDict,client)