import librosa as li
import numpy as np 
from sklearn.cluster import AffinityPropagation, KMeans
from scipy import stats

file_name = "sample.wav"
audio_time_series, sample_rate = li.load(file_name)
length_series = len(audio_time_series)
print(length_series)

zero_crossings = []
energy = []
entropy_of_energy = []
mfcc = []
chroma_stft = []
for i in range(0,length_series,int(sample_rate/5.0)):
     frame_self = audio_time_series[i:i+int(sample_rate/5.0):1]
     z = li.zero_crossings(frame_self)
     arr = np.nonzero(z)
     zero_crossings.append(len(arr[0]))
     e = li.feature.rmse(frame_self)
     energy.append(np.mean(e))
     ent = 0.0
     m = np.mean(e)
     for j in range(0,len(e[0])):
          q = np.absolute(e[0][j] - m)
          ent = ent + (q * np.log10(q))
     entropy_of_energy.append(ent)
     mt = []
     mf = li.feature.mfcc(frame_self)
     for k in range(0,len(mf)):
          mt.append(np.mean(mf[k]))
     mfcc.append(mt)
     ct = []
     cf = li.feature.chroma_stft(frame_self)
     for k in range(0,len(cf)):
          ct.append(np.mean(cf[k]))
     chroma_stft.append(ct)
     print(i)
f_list_1 = []
f_list_1.append(zero_crossings)
f_list_1.append(energy)
f_list_1.append(entropy_of_energy)
f_np_1 = np.array(f_list_1)
f_np_1 = np.transpose(f_np_1)

sp_centroid = []
sp_bandwidth = []
sp_contrast = []
sp_rolloff = []
for i in range(0,length_series,int(sample_rate/5.0)):
     frame_self = audio_time_series[i:i+int(sample_rate/5.0):1]
     cp = li.feature.spectral_centroid(y=frame_self, hop_length=220500)
     sp_centroid.append(cp[0][0])
     bp = li.feature.spectral_bandwidth(y=frame_self, hop_length=220500)
     sp_bandwidth.append(bp[0][0])
     csp = li.feature.spectral_contrast(y=frame_self, hop_length=220500)
     sp_contrast.append(np.mean(csp))
     rsp = li.feature.spectral_rolloff(y=frame_self, hop_length=220500)
     sp_rolloff.append(np.mean(rsp[0][0]))
     print(i)

f_list_2 = []
f_list_2.append(sp_centroid)
f_list_2.append(sp_bandwidth)
f_list_2.append(sp_contrast)
f_list_2.append(sp_rolloff)
f_np_2 = np.array(f_list_2)
f_np_2 = np.transpose(f_np_2)

f_np_3 = np.array(mfcc)
f_np_4 = np.array(chroma_stft)

master = np.concatenate([f_np_1, f_np_2, f_np_3, f_np_4], axis=1)

#cluster_obj = AffinityPropagation().fit(master)
cluster_obj = KMeans(n_clusters = 2 ,random_state=0).fit(master)
#print("Number of clusters : " + str(len(cluster_obj.cluster_centers_indices_)))
res = cluster_obj.predict(master)
#print(cluster_obj.get_params())
s = res[0]
t=0.0
time = []
speaker = []
time.append(t)
speaker.append(s)
for u in range(0, len(res), 1):
     if(res[u]==s):
          t=t+0.2
     else:
          t=t+0.2
          s=res[u]
          speaker.append(s)
          time.append(t)

print(time)
print(speaker)
speakerN = speaker
speakerN.append(0)
for i in range(2, len(time)):
     if((time[i]-time[i-1]) < 0.75):
          pass
     else:
          speaker[i-1] = speakerN[i-2]     

fin = []
for i in range(1,len(time)):
     if(speaker[i]!=speaker[i-1]):
          fin.append([time[i-1], speaker[i-1]])
     else:
          pass

for p in range(0, len(fin)):
     print("TIME : " + str(fin[p][0]) + " ---- " + "SPEAKER : " + str(fin[p][1]))