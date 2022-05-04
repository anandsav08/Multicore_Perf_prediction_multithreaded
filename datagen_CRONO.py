import subprocess
import re
import csv

CRONO_PATH = "/home/ak8288/multicore/final_project/crono/CRONO"
class dataWriter:
    def __init__(self,headers):
        self.writer = None
        self.headers = headers
        self.outputfile = "data_CRONO.csv"
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
        #self.TASKS = ["blackscholes","bodytrack","canneal","facesim","ferret","fluidanimate","freqmine","raytrace","streamcluster","swaptions","vips","x264"]
        # self.TASKS = ["blackscholes"]
        self.size_dict = {'simsmall' : ' 4096 16 ', 'simmedium':' 8192 32 ', 'simlarge': ' 16384 64 '}
        self.TASKS = {
                      "apsp": CRONO_PATH+"/apps/apsp/./apsp",
                      "bc": CRONO_PATH+"/apps/bc/./bc",
                      "community":CRONO_PATH+"/apps/community/./community_lock 0",
                      "connected_components":CRONO_PATH+"/apps/connected_components/./connected_components_lock 0",
                      "pagerank":CRONO_PATH+"/apps/pagerank/./pagerank 0",
                      "sssp":CRONO_PATH+"/apps/sssp/./sssp 0"
                      }
                    #   "tsp":"/home/tfb9946/multicore/CRONO/apps/tsp/./tsp {} 16"}
        
        self.THREADS = [1,2,4,8,16,32,64,128]     #wide range of threads to show increase and decrease in perf with too many threads than required
        self.data_writer = None
        self.CMD = "perf stat -o temp.txt -r 3 -e "                     #Run 3 times and take average
        #self.simSize = ["simsmall","simmedium","simlarge"]
        self.taskData = {}
        self.runs = 3                                       # Run each task 3 times and take average

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
        count = 0
        index = 0
        print("================================> RUNNING WORKLOADS <START>")
        for task in self.TASKS:
            for size in self.size_dict:
                for thread in self.THREADS:
                    try:
                        print("[ RUNNING WORKLOAD ] {} with thread {} <START>".format(task,thread))
                        localCMD = self.CMD
                        parsecCMD = " " + self.TASKS[task]
                        if(task == "community"):
                            parsecCMD += " " + str(thread) + " 5" + self.size_dict[size]
                        else:
                            parsecCMD += " " + str(thread) + self.size_dict[size]
                        localCMD = localCMD + parsecCMD
                        #print("localcmd: ",localCMD)
                        result = subprocess.run(localCMD,shell=True,stdout=subprocess.PIPE)
                        result = result.stdout.decode("utf-8")
                        self.parseResult(result,task,size,thread,index)
                        print("[ FINISHED WORKLOAD ] {} with Problem size {} and thread {} <FINISH>".format(task,size,thread))
                    except Exception as e :
                        count += 1
                        print("[EXCEPTION] ",task,size,thread)
                        print(e)
                        continue
        print("================================> RUNNING WORKLOADS <FINISH>")
        print("Exception occured {} times".format(count))

    # def runWorkload(self):
    #     print("================================> RUNNING WORKLOADS <START>")
    #     for task in self.TASKS:
    #         for size in self.simSize:
    #             for thread in self.THREADS:
    #                 print("[ RUNNING WORKLOAD ] {} with Problem size {} and thread {} <START>".format(task,size,thread))
    #                 localCMD = self.CMD
    #                 parsecCMD = " parsecmgmt -a run -p {} -n {} -i {}".format(task,thread,size)
    #                 localCMD = localCMD + parsecCMD
    #                 print("localcmd: ",localCMD)
    #                 result = subprocess.run(localCMD,shell=True,stdout=subprocess.PIPE)
    #                 result = result.stdout.decode("utf-8")
    #                 self.parseResult(result,task,size,thread)
    #                 print("[ FINISHED WORKLOAD ] {} with Problem size {} and thread {} <FINISH>".format(task,size,thread))
    #     print("================================> RUNNING WORKLOADS <FINISH>")


        

    def parseResult(self,result,task,size,thread,index):
        #parse temp.txt first for perf data
        #index +=1
        print(result)
        perfData = open("temp.txt","r")     # perf data for each innermost iteration in def runWorkload() is saved in "temp.txt" file
        lines = perfData.readlines()
        row = ""
        for line in lines:
            if("started" in line):
                continue
            if("#" not in line):
                continue
            line = line.strip()             # remove leading and trailing white spaces
            line = re.sub(' +',' ',line)    # replace multiple continuous white space with a single white space
            cols = line.split(" ")          # get each col in cols
            val = re.sub(",","",cols[0])
            if(val[0] == '#'):
                continue
            row = row + val + ","             # perf-list entries and
        # print("row I: "+row)

        #Parse the parsecmgmt command result now for real time
        lines = result.split("\n")
        run_time = 0
        for line in lines:
            line = line.strip()             # remove leading and trailing white spaces
            line = re.sub(' +',' ',line)    # replace multiple continuous white space with a single white space
            if("Time" in line or "time" in line):
                cols = line.split(":")
                cols = [x.strip(' ') for x in cols]
                cur_time = float(cols[1].split(" ")[0])
                print("REAL TIME ====> ",cur_time)

                run_time = run_time + cur_time
        average_run_time = run_time / self.runs     # Take average of run times
        speedup = 1.0
        if(thread==1):
            if(task not in self.taskData):
                self.taskData[task] = {}
            self.taskData[task][size] = average_run_time    # save base time for 1 thread for each problem size
        else:
            speedup = self.taskData[task][size] / average_run_time

        row += size + "," + str(thread) + "," + str(average_run_time) + "," + str(speedup) + "," + str(task) +"\n"
        # print("row II: "+row)
        #print("Row {}: {}\n".format(index,row))
        self.data_writer.writeRow(row)
        perfData.close()

if __name__ == "__main__":
    data = Data()
    data.getPerformanceList()
    data.initializeWriter()
    data.writeHeader()
    data.buildPerfCMD()
    #data.buildWorkloads()
    data.runWorkload()
    data.closeWriter()
