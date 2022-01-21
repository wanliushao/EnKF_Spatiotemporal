#This program test MPI-EnKF algorithm
#Right now it just for sigle grid, using LAI and also not for stochastic prior parameters
#Update 08/03/2017 Shaoqing
from mpi4py import MPI
from numpy.linalg import inv
from datetime import date
import numpy as np
import numpy.matlib
import os
from netCDF4 import Dataset
import glob
import scipy.stats as stats
from scipy.linalg import sqrtm

#print("hello world")
comm=MPI.COMM_WORLD
rank=comm.Get_rank()

recv_data = None
file1='/home/sqliu/software_install/cesm1_2_2/scripts/Mojave_2x2_precip_test'
file2='/Mojave_2x2_precip_test_trans_'
file3='/rundir'
command1='./Mojave_2x2_precip_test_trans_'
command2='.run'

EnKF_size=100;
EnKF_number=7;
para_number=4;
obs_number=3;
observation_number=3;
npros=51; #mpirun -np np programe
EnKF_time=86; #EnKF time steps
per_jobs=EnKF_size/(npros-1); #the number of jobs for each processor to handle

if __name__=="__main__":
	def main():
		if (rank>0):
			for m in range(1,EnKF_time+1):
				for p in range(5):
					subpro(rank);
					comm.Barrier();
		elif (rank==0):
			data=np.arange(npros)
			lai_sim=np.zeros((EnKF_time,EnKF_size))
			lai_obs=np.empty(0);
			sm1_obs=np.empty(0);
			sm2_obs=np.empty(0);
			run_days=np.empty(0);
			for line in open('Kelmet_lai_sm.txt'):
				line=line.rstrip('\n');
				lai_obs=np.append(lai_obs,line.split("\t")[1]);
				sm1_obs=np.append(sm1_obs,line.split("\t")[2]);
				sm2_obs=np.append(sm2_obs,line.split("\t")[3]);
				run_days=np.append(run_days,line.split("\t")[4]);
			run_days=run_days.astype(np.int);

			print("process {} scatter data {} to other processes".format(rank, data))
			#begin EnKF
			for m in range(1,EnKF_time+1):#this will be time step in future
				observation=np.matlib.zeros((3,1));
				observation[0,0]=lai_obs[m-1];
				observation[1,0]=sm1_obs[m-1];
				observation[2,0]=sm2_obs[m-1];
				observation=observation.astype(np.float);

				#change run_days first
				for j in range(1,EnKF_size+1):
					file=file1+file2+format(j, '003')
					os.chdir(file);
					fid=open('change_runtime.sh','w');
					if (m==1):
						p_rundays=8;
					else:
						p_rundays=run_days[m-2];
					fid.write(r'sed -i "62s/value=\"%d\"/value=\"%d\"/" env_run.xml' % (p_rundays,run_days[m-1])) #use ./xmlchange has some problem with xml structuce
					fid.close();
					os.system('bash change_runtime.sh');
					
				for p in range(5):
					for j in range(1,EnKF_size+1):
						file=file1+file2+format(j, '003')
						os.chdir(file)
						if (p==0) and (m>1):
							cmdstring='cp rundir/rpointer* '+file
							os.system(cmdstring)
						if (p>0):
							cmdstring='cp rpointer* rundir/'
							os.system(cmdstring)
					for i in range(1,npros):
						comm.Send(data[i-1], dest=i, tag=11)
					comm.Barrier();
					
					for j in range(1,EnKF_size+1):
						file=file1+file2+format(j, '003')
						os.chdir(file)
						if (p==4) and (m==1):
							cmdstring='sed -i "79s/FALSE/TRUE/" env_run.xml'
							os.system(cmdstring)
						if (p==0) and (m==1):
							cmdstring='cp rundir/rpointer* '+file
							os.system(cmdstring)
					#get LAI simulations and SLA, Nm and RootD parameters
					Y_f=Get_simulation(EnKF_number, para_number,EnKF_size);
					if (p<4):
						para_update=para_EnKF(observation,Y_f,EnKF_size,EnKF_number,para_number)
						#Update paras and states;assign new paras and ensembles to each folder(ensemble member)
						Update(para_update, EnKF_size,p)
					else:
						logfid=open('/home/sqliu/software_install/cesm1_2_2/scripts/Mojave_2x2_precip_test/para_0508.log','a');
						for k in range(EnKF_size):
							line=str(para_update[0,k])+'\t'+str(para_update[1,k])+'\t'+str(para_update[2,k])+'\t'+str(para_update[3,k])+'\n'
							logfid.write(line)
						logfid.close()

						state_update=state_EnKF(observation,Y_f,EnKF_size,EnKF_number,para_number)
						Update(para_update, EnKF_size,p)

						logfid=open('/home/sqliu/software_install/cesm1_2_2/scripts/Mojave_2x2_precip_test/state_0508.log','a');
						for k in range(EnKF_size):
							line=str(state_update[0,k])+'\t'+ str(state_update[1,k])+'\t'+str(state_update[2,k])+'\t'+str(observation[0,0])+'\t'+str(observation[1,0])+'\t'+str(observation[2,0])+'\n';
							logfid.write(line)
						logfid.close()
		MPI.Finalize()

	def subpro(rank):
		data=np.empty(1);
		comm.Recv(data, source=0, tag=11)
		for i in range(1,per_jobs+1):
			indx=(rank-1)*per_jobs+i;
			file=file1+file2+format(indx, '003')
			print("process {} deal with directory{}".format(rank, file))
			os.chdir(file)
			cmdstring=command1+format(indx, '003')+command2
			os.system(cmdstring)

	def Get_simulation(EnKF_number,para_number,EnKF_size):
		#para=['slatop', 'leafcn', 'rootb_par', 'flnr', 'froot_leaf', 'frootcn', 'mp_pft', 'leaf_long']
		para=['slatop','leafcn','rootb_par','leaf_long']
		Y_f=np.matlib.zeros((EnKF_number,EnKF_size))
		for j in range(1,EnKF_size+1):
			file=file1+file2+format(j, '003')+file3
			os.chdir(file)
			newest = max(glob.iglob('*clm2.r.*nc'), key=os.path.getctime)
			ncfid=Dataset(newest,'r');
			LAI=ncfid.variables['tlai'][9]*0.15; #test one grid first, at the 8th day
			SM1=ncfid.variables['H2OSOI_LIQ'][0,8];#unit kg/m2, upper 5 layers are snow
			SM2=ncfid.variables['H2OSOI_LIQ'][0,10];#unit kg/m2, upper 5 layers are snow
			ncfid.close()
			Y_f[para_number,j-1]=LAI;
			Y_f[para_number+1,j-1]=SM1;#ignore SM error first
			Y_f[para_number+2,j-1]=SM2;#ignore SM error first
			pft_file='/home/sqliu/software_install/cesm1_2_2/scripts/Mojave_2x2_precip_test/PFT_file/pft-physiology_constant_allocation_'+format(j,'003')+'.nc'
			ncfid=Dataset(pft_file,'r');
			for i in range(EnKF_number-3):
				Y_f[i,j-1]=ncfid.variables[para[i]][9];
			ncfid.close();
		return (Y_f)

	def para_EnKF(observation,simulation, EnKF_size, EnKF_number,para_number):
		observation_number=3;
		H=np.matlib.zeros((observation_number,EnKF_number));
		H[0,para_number]=1;
		H[1,para_number+1]=1;
		H[2,para_number+2]=1;
		R=np.matlib.zeros((observation_number,observation_number));
		error=np.array([0.1,2,5])
		np.fill_diagonal(R,np.square(error));
		f_mean=np.asmatrix(np.mean(simulation,axis=1));
		Ensemble_dev=simulation-np.repeat(f_mean,EnKF_size,axis=1);
		Pb=np.dot(Ensemble_dev,np.transpose(Ensemble_dev))/(EnKF_size-1);
		temp1=np.dot(np.dot(H,Pb),np.transpose(H))+ R;
		temp2=np.dot(Pb,np.transpose(H));
		K=np.dot(temp2,inv(temp1));
		u_mean=f_mean + np.dot(K,(observation-np.dot(H,f_mean)));
		
		temp1=inv(sqrtm(temp1));
		K_p=np.dot(temp2,np.transpose(temp1)) #size (EnKF_number-observation_number)xobservation_number
		K_p=np.dot(K_p,inv(inv(temp1)+np.sqrt(R))); #size (EnKF_number-observation_number)xobservation_number
		obs_perturb=np.zeros((1,EnKF_size));
		temp1=simulation-np.repeat(f_mean,EnKF_size,axis=1);#(EnKF_number-1)x(EnKF_size)
		EnKF_perturb=temp1+np.dot(K_p,(obs_perturb-np.dot(H,temp1)))#size (EnKF_number-1)xEnKF_size
		Y_update=np.repeat(u_mean,EnKF_size,axis=1)+EnKF_perturb;#size (EnKF_number-1)xEnKF_size igma=0.1*1.2;
		return (Y_update)

	def state_EnKF(observation,simulation, EnKF_size, EnKF_number,para_number):
		observation_number=3;
		s_EnKF_number = observation_number;
		H=np.matlib.zeros((observation_number,s_EnKF_number));
		H[0,0]=1;
		H[1,1]=1;
		H[2,2]=1;
		R=np.matlib.zeros((observation_number,observation_number));
		error=np.array([0.1,2,5])
		np.fill_diagonal(R,np.square(error));
		simulation = simulation[para_number:EnKF_number,:]
		f_mean=np.asmatrix(np.mean(simulation,axis=1)); # size (EnKF_number-para_number) x 1
		Ensemble_dev=simulation-np.repeat(f_mean,EnKF_size,axis=1);
		Ensemble_dev=Ensemble_dev #size (EnKF_number-para_number)xEnKF_size
		Pb=np.dot(Ensemble_dev,np.transpose(Ensemble_dev))/(EnKF_size-1); #size (EnKF_number-para_number)x(EnKF_number-para_number)
		temp1=np.dot(np.dot(H,Pb),np.transpose(H))+ R; #size observation_number x obs_number
		temp2=np.dot(Pb,np.transpose(H)); #size observation_number x obs_number
		K=np.dot(temp2,inv(temp1)); #size obs_number x obs_number
		u_mean=f_mean + np.dot(K,(observation-np.dot(H,f_mean))); #size obs_number x 1
		
		temp1=inv(sqrtm(temp1));
		K_p=np.dot(temp2,np.transpose(temp1)) #size observation_number x observation_number
		K_p=np.dot(K_p,inv(inv(temp1)+np.sqrt(R))); #size observation_number x observation_number
		obs_perturb=np.zeros((1,EnKF_size));
		temp1=simulation-np.repeat(f_mean,EnKF_size,axis=1);#obs_number x (EnKF_size)
		EnKF_perturb=temp1+np.dot(K_p,(obs_perturb-np.dot(H,temp1)))#size obs_number x EnKF_size
		Y_update=np.repeat(u_mean,EnKF_size,axis=1)+EnKF_perturb;#size obs_number x EnKF_size;
		return (Y_update)

	def Update(updates,EnKF_size,p):
		#para=['slatop', 'leafcn', 'rootb_par', 'flnr', 'froot_leaf', 'frootcn', 'mp_pft', 'leaf_long']
		#stdvs=[0.01,20,1.5,0.025,0.5,20,5,0.5]
		para=['slatop','leafcn','rootb_par','leaf_long']
		stdvs = [0.01,20,1.5,0.5]
		updates_mean=np.squeeze(np.asarray(np.mean(updates[0:para_number,:],axis=1)));
		for j in range(1,EnKF_size+1):
			file=file1+file2+format(j, '003')+file3
			os.chdir(file)
			newest = max(glob.iglob('*clm2.r.*nc'), key=os.path.getctime) #this is updated lai, CLM start next ensemble run based on this restart file
			if (p==4):
				ncfid=Dataset(newest,'r+');
				ncfid.variables['tlai'][9]=updates[para_number,j-1]/0.15; #BES fraction in one grid cell 15%
				ncfid.variables['H2OSOI_LIQ'][0,8]=updates[para_number+1,j-1];
				ncfid.variables['H2OSOI_LIQ'][0,10]=updates[para_number+2,j-1]
				ncfid.close();
			pft_file='/home/sqliu/software_install/cesm1_2_2/scripts/Mojave_2x2_precip_test/PFT_file/pft-physiology_constant_allocation_'+format(j,'003')+'.nc'
			ncfid=Dataset(pft_file,'r+');
			for i in range(EnKF_number-3):
				#if (p==4):
				#print ("ddd=",updates_stde[i])
				ncfid.variables[para[i]][9]=0.99*updates[i,j-1]+0.01*updates_mean[i]+np.random.normal(0,0.1*stdvs[i],1);
				#else:
				#	ncfid.variables[para[i]][9]=updates[i,j-1]
			ncfid.close();
	main();
