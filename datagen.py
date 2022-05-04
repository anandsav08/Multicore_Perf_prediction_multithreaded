import subprocess
import re
import csv

class dataWriter:
    def __init__(self,headers):
        self.writer = None
        self.headers = headers
        self.outputfile = "data.csv"
        self.file = None

    def initializeWriter(self):
        self.file = open(self.outputfile,"w")
        self.writer = csv.writer(self.file)

    def writeHeader(self):
        self.writer = csv.DictWriter(self.file,delimiter=",",fieldnames=self.headers)
        self.writer.writeheader()
    
    def writeRow(self,line):
        self.file.write(line)
    
    def closeWriter(self):
        self.file.close()

class Data:

    def __init__(self):
        self.PERF_LIST = []                                 # list of performane counters from "perf" linux command
        self.TASKS = ["blackscholes","bodytrack","canneal","facesim","ferret","fluidanimate","freqmine","raytrace","streamcluster","swaptions","vips","x264"]
        self.TASKS_SPLASH = ["splash2x.fft","splash2x.raytrace","splash2x.barnes","splash2x.fmm","splash2x.lu_cb","splash2x.lu_ncb","splash2x.ocean_cp","splash2x.radiosity","splash2x.radix","splash2x.volrend","splash2x.water_spatial"]
        # self.TASKS = ["blackscholes"]
        self.THREADS = [1,2,4,8,16,32,64,128]     #wide range of threads to show increase and decrease in perf with too many threads than required
        self.data_writer = None
        self.CMD = "perf stat -o temp.txt -r 3 -e "                     #Run 3 times and take average
        self.simSize = ["simsmall","simmedium","simlarge"]
        self.taskData = {}
        self.runs = 3                                       # Run each task 3 times and take average

    def buildSplash(self):
        print("[ BUILDING WORKLOADS STARTED ]")
        for task in self.TASKS_SPLASH:
            print("Building {}".format(task))
            subprocess.run(["parsecmgmt","-a","build","-p",task],stdout=subprocess.PIPE)
        print("[ BUILDING WORKLOADS FINISHED ]")

    def getPerformanceList(self):
        result = subprocess.run(['perf', 'list'], stdout=subprocess.PIPE)
        perf_counters = result.stdout.decode("utf-8")       #decode to string readable format
        perf_counters = perf_counters.split("\n")
        keywords = ["Hardware event","Software Event"]
        for perf_counter in perf_counters:
            line = perf_counter
            line = line.strip()                             #remove leading and trailing whitespaces
            line = re.sub(' +',' ',line)
            cols = line.split(" ")  
            if("Software event" in line):
                self.PERF_LIST.append(cols[0])
            if("Hardware event" in line):
                self.PERF_LIST.append(cols[0])
            if("Hardware cache event" in line):
                self.PERF_LIST.append(cols[0])
        self.PERF_LIST = self.PERF_LIST[0:35]
        extras = ["problem_size","threads","time","speedup"]
        self.PERF_LIST.extend(extras)
        
    def initializeWriter(self):
        self.data_writer = dataWriter(self.PERF_LIST)
        self.data_writer.initializeWriter()

    def writeHeader(self):
        self.data_writer.writeHeader()

    def closeWriter(self):
        self.data_writer.closeWriter()

    def buildWorkloads(self):
        print("[ BUILDING WORKLOADS STARTED ]")
        for task in self.TASKS:
            print("Building {}".format(task))
            subprocess.run(["parsecmgmt","-a","build","-p",task],stdout=subprocess.PIPE)
        print("[ BUILDING WORKLOADS FINISHED ]")

    def buildPerfCMD(self):
        perflist = ""
        for perf in self.PERF_LIST[:-4]:
            perflist = perflist + perf + ","
        perflist = perflist[:-1]
        self.CMD = self.CMD + perflist

    def runWorkload(self):
        print("================================> RUNNING WORKLOADS <START>")
        for task in self.TASKS:
            for size in self.simSize:
                for thread in self.THREADS:
                    print("[ RUNNING WORKLOAD ] {} with Problem size {} and thread {} <START>".format(task,size,thread))    
                    localCMD = self.CMD
                    parsecCMD = " parsecmgmt -a run -p {} -n {} -i {}".format(task,thread,size)
                    localCMD = localCMD + parsecCMD
                    # print("localcmd: ",localCMD)
                    result = subprocess.run(localCMD,shell=True,stdout=subprocess.PIPE)
                    result = result.stdout.decode("utf-8")
                    self.parseResult(result,task,size,thread)
                    print("[ FINISHED WORKLOAD ] {} with Problem size {} and thread {} <FINISH>".format(task,size,thread))   
        print("================================> RUNNING WORKLOADS <FINISH>")

    def runSplashWorkLoad(self):
        print("================================> RUNNING WORKLOADS <START>")
        for task in self.TASKS_SPLASH:
            for size in self.simSize:
                for thread in self.THREADS:
                    print("[ RUNNING WORKLOAD  ] {} with Problem size {} and thread {} <START>".format(task,size,thread))    
                    localCMD = self.CMD
                    parsecCMD = " parsecmgmt -a run -p {} -n {} -i {}".format(task,thread,size)
                    localCMD = localCMD + parsecCMD
                    # print("localcmd: ",localCMD)
                    result = subprocess.run(localCMD,shell=True,stdout=subprocess.PIPE)
                    result = result.stdout.decode("utf-8")
                    self.parseResult(result,task,size,thread)
                    print("[ FINISHED WORKLOAD ] {} with Problem size {} and thread {} <FINISH>".format(task,size,thread))   
        print("================================> RUNNING WORKLOADS <FINISH>")

    def parseResult(self,result,task,size,thread):
        #parse temp.txt first for perf data 
        perfData = open("temp.txt","r")     # perf data for each innermost iteration in def runWorkload() is saved in "temp.txt" file
        lines = perfData.readlines()
        row = ""
        perf_count = 0
        for line in lines:
            if("started" in line):
                continue
            if("#" not in line):
                continue
            if(perf_count == 35):
                break
            line = line.strip()             # remove leading and trailing white spaces
            line = re.sub(' +',' ',line)    # replace multiple continuous white space with a single white space
            cols = line.split(" ")          # get each col in cols
            val = re.sub(",","",cols[0])
            if(val[0] == '#'):
                continue
            row = row + val + ","             # perf-list entries and 
            perf_count+=1
        # print("row I: "+row)

        #Parse the parsecmgmt command result now for real time
        lines = result.split("\n") 
        run_time = 0
        for line in lines:
            line = line.strip()             # remove leading and trailing white spaces
            line = re.sub(' +',' ',line)    # replace multiple continuous white space with a single white space
            if("real" in line):
                cols = line.split("\t") 
                time_string = cols[1]
                minutes = int(time_string.split("m")[0])
                seconds = float(time_string.split("m")[1].split("s")[0])
                cur_time = minutes*60+seconds
                # print("REAL TIME ====> ",cur_time)
                run_time = run_time + cur_time
        average_run_time = run_time / self.runs     # Take average of run times
        speedup = 1.0
        if(thread==1):
            if(task not in self.taskData):
                self.taskData[task] = {}
            self.taskData[task][size] = average_run_time    # save base time for 1 thread for each problem size
        else:
            speedup = self.taskData[task][size] / average_run_time

        row += size + "," + str(thread) + "," + str(average_run_time) + "," + str(speedup) +"\n"
        # print("row II: "+row)
        self.data_writer.writeRow(row)
        perfData.close()

if __name__ == "__main__":
    data = Data()
    data.getPerformanceList()
    data.initializeWriter()
    data.writeHeader()
    data.buildPerfCMD()
    
    #PARSEC
    # data.buildWorkloads()
    # data.runWorkload()

    #SPLASH
    # data.buildSplash()
    data.runSplashWorkLoad()

    
    data.closeWriter()
    