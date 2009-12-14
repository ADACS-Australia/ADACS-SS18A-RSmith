% This code reads in a list of found injections, a list of missed injections,
%  and a list of vetoes (if required). Missed / found plots (altogether and
%  individually are made. Also produced is ths efficiency curve for triples.
% Please note that the  distance range is hard-coded in the efficiency calculation.
%
% Usage example:
% >> found_list=['injH1H2L1coincs1001.xml';'injH1H2L1coincs1002.xml';'injH1H2L1coincs1003.xml'];
% >> missed_list=['injH1H2L1missed1001.xml';'injH1H2L1missed1002.xml';'injH1H2L1missed1002.xml'];
% >> veto_list = [];
% >> inj_type = ['EOBNR']; ( other types can be 'PHENOM', 'RINGDOWN', or 'ALL')
% >> veto_type = ['NOVETO']; or CAT2, CAT23
% >> plottrigs( inj_type, veto_type, found_list,missed_list,veto_list )


function plottrigs( inj_type, veto_type, found_list,missed_list,veto_list )

% read in the injection lists
eval(['injH1t=load(''' veto_type '_inj_' inj_type '_H1trip.mat'');'])
eval(['injL1t=load(''' veto_type '_inj_' inj_type '_L1trip.mat'');'])
eval(['injH2t=load(''' veto_type '_inj_' inj_type '_H2trip.mat'');'])
eval(['injH1inL1d=load(''' veto_type '_inj_' inj_type '_H1inL1doub.mat'');'])
eval(['injL1inH1d=load(''' veto_type '_inj_' inj_type '_L1inH1doub.mat'');'])
eval(['injH1inH2d=load(''' veto_type '_inj_' inj_type '_H1inH2doub.mat'');'])
eval(['injH2inH1d=load(''' veto_type '_inj_' inj_type '_H2inH1doub.mat'');'])
eval(['injL1inH2d=load(''' veto_type '_inj_' inj_type '_L1inH2doub.mat'');'])
eval(['injH2inL1d=load(''' veto_type '_inj_' inj_type '_H2inL1doub.mat'');'])

% if no cut
injH1inL1dcut.ind = [];
injH1inH2dcut.ind = [];
injH2inL1dcut.ind = [];
injH1inL1daftercut.ind = injH1inL1d.ind;
injH1inH2daftercut.ind = injH1inH2d.ind;
injH2inL1daftercut.ind = injH2inL1d.ind;

% make veto files into .mat files
if length(veto_list)>0 && exist('H1vetolist.mat','file')==0
  fprintf('generating veto lists\n   This might take a while.....\n');
  [num,H1vetostart,H1vetoend,duration]=textread(veto_list(1,:),'%d %d %d %d','commentstyle','shell');
  H1vetolist=[];
  % make a list of all the gps seconds vetoes
  j=1;
  for i=1:length(H1vetostart)
    H1vetolist=[H1vetolist, H1vetostart(i):H1vetoend(i)];
  end
  save H1vetolist.mat H1vetolist
  fprintf('H1 veto list saved\n')

  [num,H2vetostart,H2vetoend,duration]=textread(veto_list(2,:),'%d %d %d %d','commentstyle','shell');
  H2vetolist=[];
  j=1;
  for i=1:length(H2vetostart)
    H2vetolist=[H2vetolist, H2vetostart(i):H2vetoend(i)];
  end
  save H2vetolist.mat H2vetolist
  fprintf('H2 veto list saved\n')

  [num,L1vetostart,L1vetoend,duration]=textread(veto_list(3,:),'%d %d %d %d','commentstyle','shell');
  L1vetolist=[];
  j=1;
  for i=1:length(L1vetostart)
    L1vetolist=[L1vetolist, L1vetostart(i):L1vetoend(i)];
  end
  save L1vetolist.mat L1vetolist
  fprintf('L1 veto list saved\n')
end

%=========================== Injected Quantities ===========================%

% Read in missed and found sim_ringdown tables
N_files = length(found_list)

eval(['fsim=readMeta(found_list{1},''sim_ringdown'',0,''h_start_time,h_start_time_ns,l_start_time,l_start_time_ns,mass,spin,frequency,quality,eff_dist_h,eff_dist_l,distance,hrss_h,hrss_l'');'])

eval(['msim=readMeta(missed_list{1},''sim_ringdown'',0,''h_start_time,h_start_time_ns,l_start_time,l_start_time_ns,mass,spin,frequency,quality,eff_dist_h,eff_dist_l,distance,hrss_h,hrss_l'');'])
A=1:N_files
for k=1:length(fsim.mass)
  fsim.run(k)=A(1);
end
fsim.run=transpose(fsim.run);

for k=1:length(msim.mass)
  msim.run(k)=A(1);
end
msim.run=transpose(msim.run);

for i=2:N_files
eval(['fsimi=readMeta(found_list{i},''sim_ringdown'',0,''h_start_time,h_start_time_ns,l_start_time,l_start_time_ns,mass,spin,quality,frequency,eff_dist_h,eff_dist_l,distance,hrss_h,hrss_l'');'])
eval(['msimi=readMeta(missed_list{i},''sim_ringdown'',0,''h_start_time,h_start_time_ns,l_start_time,l_start_time_ns,mass,spin,quality,frequency,eff_dist_h,eff_dist_l,distance,hrss_h,hrss_l'');'])

  for k=1:length(fsimi.quality)
    fsimi.run(k)=A(i);
  end
  fsimi.run=transpose(fsimi.run);

  for k=1:length(msimi.quality)
    msimi.run(k)=A(i);
  end
  msimi.run=transpose(msimi.run);

  fsim.h_start_time=[fsim.h_start_time;fsimi.h_start_time];
  fsim.h_start_time_ns=[fsim.h_start_time_ns;fsimi.h_start_time_ns];
  fsim.l_start_time=[fsim.l_start_time;fsimi.l_start_time];
  fsim.l_start_time_ns=[fsim.l_start_time_ns;fsimi.l_start_time_ns];
  fsim.frequency=[fsim.frequency;fsimi.frequency];
  fsim.quality=[fsim.quality;fsimi.quality];
  fsim.mass=[fsim.mass;fsimi.mass];  
  fsim.spin=[fsim.spin;fsimi.spin];
  fsim.eff_dist_h=[fsim.eff_dist_h;fsimi.eff_dist_h];
  fsim.eff_dist_l=[fsim.eff_dist_l;fsimi.eff_dist_l];
  fsim.distance=[fsim.distance;fsimi.distance];
  fsim.run=[fsim.run;fsimi.run];
  fsim.hrss_h=[fsim.hrss_h;fsimi.hrss_h];
  fsim.hrss_l=[fsim.hrss_l;fsimi.hrss_l];
  msim.h_start_time=[msim.h_start_time;msimi.h_start_time];
  msim.h_start_time_ns=[msim.h_start_time_ns;msimi.h_start_time_ns];
  msim.l_start_time=[msim.l_start_time;msimi.l_start_time];
  msim.l_start_time_ns=[msim.l_start_time_ns;msimi.l_start_time_ns];
  msim.frequency=[msim.frequency;msimi.frequency];
  msim.quality=[msim.quality;msimi.quality];
  msim.mass=[msim.mass;msimi.mass];           
  msim.spin=[msim.spin;msimi.spin];
  msim.eff_dist_h=[msim.eff_dist_h;msimi.eff_dist_h];
  msim.eff_dist_l=[msim.eff_dist_l;msimi.eff_dist_l];
  msim.distance=[msim.distance;msimi.distance];
  msim.run=[msim.run;msimi.run];
  msim.hrss_h=[msim.hrss_h;msimi.hrss_h];
  msim.hrss_l=[msim.hrss_l;msimi.hrss_l];

end

msim.th=msim.h_start_time+msim.h_start_time_ns/1e9;
msim.tl=msim.l_start_time+msim.l_start_time_ns/1e9;
msim.f=msim.frequency;
msim.q=msim.quality;
msim.m=msim.mass;
msim.a=msim.spin;
msim.d=msim.distance;
msim.dh=msim.eff_dist_h;
msim.dl=msim.eff_dist_l;

fsim.th=fsim.h_start_time+fsim.h_start_time_ns/1e9;
fsim.tl=fsim.l_start_time+fsim.l_start_time_ns/1e9;
fsim.f=fsim.frequency;
fsim.q=fsim.quality;
fsim.m=fsim.mass;
fsim.a=fsim.spin;
fsim.d=fsim.distance;
fsim.dh=fsim.eff_dist_h;
fsim.dl=fsim.eff_dist_l;

for i=1:length(fsim.th)
  fsim.ind(i)=i;
end

% In separate.m we assign a unique index to each coincidence, which
% maps over to the fsim array (assuming we have read the sim files in 
% in the same order as we did the sngl files in separate.m).
fsimH1H2L1.ind=fsim.ind(injH1t.ind);
fsimH1H2L1.f=transpose(fsim.f(injH1t.ind));
fsimH1H2L1.q=transpose(fsim.q(injH1t.ind));
fsimH1H2L1.m=transpose(fsim.m(injH1t.ind));
fsimH1H2L1.a=transpose(fsim.a(injH1t.ind));
fsimH1H2L1.d=transpose(fsim.d(injH1t.ind));
fsimH1H2L1.dh=transpose(fsim.dh(injH1t.ind));
fsimH1H2L1.dl=transpose(fsim.dl(injH1t.ind));
fsimH1H2L1.th=transpose(fsim.th(injH1t.ind));
fsimH1H2L1.tl=transpose(fsim.tl(injH1t.ind));
fsimH1H2L1.run=transpose(fsim.run(injH1t.ind));
fsimH1H2L1.hrss_h=transpose(fsim.hrss_h(injH1t.ind));
fsimH1H2L1.hrss_l=transpose(fsim.hrss_l(injH1t.ind));
eval(['save ' veto_type '_' inj_type '_simH1H2L1.mat -struct fsimH1H2L1'])

fsimH1L1d.ind=fsim.ind(injH1inL1daftercut.ind);
fsimH1L1d.f=transpose(fsim.f(injH1inL1daftercut.ind));
fsimH1L1d.q=transpose(fsim.q(injH1inL1daftercut.ind));
fsimH1L1d.m=transpose(fsim.m(injH1inL1daftercut.ind));
fsimH1L1d.a=transpose(fsim.a(injH1inL1daftercut.ind));
fsimH1L1d.d=transpose(fsim.d(injH1inL1daftercut.ind));
fsimH1L1d.dh=transpose(fsim.dh(injH1inL1daftercut.ind));
fsimH1L1d.dl=transpose(fsim.dl(injH1inL1daftercut.ind));
fsimH1L1d.th=transpose(fsim.th(injH1inL1daftercut.ind));
fsimH1L1d.tl=transpose(fsim.tl(injH1inL1daftercut.ind));
fsimH1L1d.run=transpose(fsim.run(injH1inL1daftercut.ind));
fsimH1L1d.hrss_h=transpose(fsim.hrss_h(injH1inL1daftercut.ind));
fsimH1L1d.hrss_l=transpose(fsim.hrss_l(injH1inL1daftercut.ind));
eval(['save  ' veto_type '_' inj_type '_simH1L1.mat -struct fsimH1L1d'])

fsimH1H2d.ind=fsim.ind(injH1inH2daftercut.ind);
fsimH1H2d.f=transpose(fsim.f(injH1inH2daftercut.ind));
fsimH1H2d.q=transpose(fsim.q(injH1inH2daftercut.ind));
fsimH1H2d.m=transpose(fsim.m(injH1inH2daftercut.ind));
fsimH1H2d.a=transpose(fsim.a(injH1inH2daftercut.ind));
fsimH1H2d.d=transpose(fsim.d(injH1inH2daftercut.ind));
fsimH1H2d.dh=transpose(fsim.dh(injH1inH2daftercut.ind));
fsimH1H2d.dl=transpose(fsim.dl(injH1inH2daftercut.ind));
fsimH1H2d.th=transpose(fsim.th(injH1inH2daftercut.ind));
fsimH1H2d.tl=transpose(fsim.tl(injH1inH2daftercut.ind));
fsimH1H2d.run=transpose(fsim.run(injH1inH2daftercut.ind));
fsimH1H2d.hrss_h=transpose(fsim.hrss_h(injH1inH2daftercut.ind));
fsimH1H2d.hrss_l=transpose(fsim.hrss_l(injH1inH2daftercut.ind));
eval(['save  ' veto_type '_' inj_type '_simH1H2.mat -struct fsimH1H2d'])

fsimL1H2d.ind=fsim.ind(injH2inL1daftercut.ind);
fsimL1H2d.f=transpose(fsim.f(injH2inL1daftercut.ind));
fsimL1H2d.q=transpose(fsim.q(injH2inL1daftercut.ind));
fsimL1H2d.m=transpose(fsim.m(injH2inL1daftercut.ind));
fsimL1H2d.a=transpose(fsim.a(injH2inL1daftercut.ind));
fsimL1H2d.d=transpose(fsim.d(injH2inL1daftercut.ind));
fsimL1H2d.dh=transpose(fsim.dh(injH2inL1daftercut.ind));
fsimL1H2d.dl=transpose(fsim.dl(injH2inL1daftercut.ind));
fsimL1H2d.th=transpose(fsim.th(injH2inL1daftercut.ind));
fsimL1H2d.tl=transpose(fsim.tl(injH2inL1daftercut.ind));
fsimL1H2d.run=transpose(fsim.run(injH2inL1daftercut.ind));
fsimL1H2d.hrss_h=transpose(fsim.hrss_h(injH2inL1daftercut.ind));
fsimL1H2d.hrss_l=transpose(fsim.hrss_l(injH2inL1daftercut.ind));
eval(['save ' veto_type '_' inj_type '_simH2L1.mat -struct fsimL1H2d'])

foundd=[fsimH1L1d.ind,fsimH1H2d.ind,fsimL1H2d.ind];
threshcut.ind=fsim.ind([injH1inH2dcut.ind,injH2inL1dcut.ind,injH1inL1dcut.ind]);
threshcut.f=fsim.f([injH1inH2dcut.ind,injH2inL1dcut.ind,injH1inL1dcut.ind]);
threshcut.d=fsim.d([injH1inH2dcut.ind,injH2inL1dcut.ind,injH1inL1dcut.ind]);
threshcut.dh=fsim.dh([injH1inH2dcut.ind,injH2inL1dcut.ind,injH1inL1dcut.ind]);
threshcut.dl=fsim.dl([injH1inH2dcut.ind,injH2inL1dcut.ind,injH1inL1dcut.ind]);
threshcut.th=fsim.th([injH1inH2dcut.ind,injH2inL1dcut.ind,injH1inL1dcut.ind]);
threshcut.tl=fsim.tl([injH1inH2dcut.ind,injH2inL1dcut.ind,injH1inL1dcut.ind]);
threshcut.run=fsim.run([injH1inH2dcut.ind,injH2inL1dcut.ind,injH1inL1dcut.ind]);
threshcut.hrss_h=fsim.hrss_h([injH1inH2dcut.ind,injH2inL1dcut.ind,injH1inL1dcut.ind]);
threshcut.hrss_l=fsim.hrss_l([injH1inH2dcut.ind,injH2inL1dcut.ind,injH1inL1dcut.ind]);

msim.th=[msim.th;threshcut.th];
msim.tl=[msim.tl;threshcut.tl];
msim.f=[msim.f;threshcut.f];
msim.d=[msim.d;threshcut.d];
msim.dh=[msim.dh;threshcut.dh];
msim.dl=[msim.dl;threshcut.dl];
msim.run=[msim.run;threshcut.run];
msim.hrss_h=[msim.hrss_h;threshcut.hrss_h];
msim.hrss_l=[msim.hrss_l;threshcut.hrss_l];

% read in Veto files
if length(veto_list)>0
  A=load('H1vetolist.mat');
  H1vetolist=A.H1vetolist;
  B=load('H2vetolist.mat');
  H2vetolist=B.H2vetolist;
  C=load('L1vetolist.mat');
  L1vetolist=C.L1vetolist;
else
  H1vetolist=[];
  H2vetolist=[];
  L1vetolist=[];
end

% times vetoed in all three ifos
vetoedinH1H2L1=intersect(H1vetolist,intersect(H2vetolist,L1vetolist));
% times vetoed in two ifos (only)
vetoedinH1H2=setdiff(intersect(H1vetolist,H2vetolist),vetoedinH1H2L1);
vetoedinH1L1=setdiff(intersect(H1vetolist,L1vetolist),vetoedinH1H2L1);
vetoedinL1H2=setdiff(intersect(L1vetolist,H2vetolist),vetoedinH1H2L1);
% time vetoed in one ifo (only)
vetoedinH1=setdiff(H1vetolist,union(H2vetolist,L1vetolist));
vetoedinH2=setdiff(H2vetolist,union(H1vetolist,L1vetolist));
vetoedinL1=setdiff(L1vetolist,union(H2vetolist,H1vetolist));

% total triple time lost
losttriptimes=[vetoedinH1H2L1,vetoedinH1H2,vetoedinH1L1,vetoedinL1H2,vetoedinH1,vetoedinH2,vetoedinL1];

% list the injections that were missed during these times
vetot.t=transpose(msim.th(ismember(floor(msim.th),losttriptimes)));
vetot.d=transpose(msim.d(ismember(floor(msim.th),losttriptimes)));
vetot.dh=transpose(msim.dh(ismember(floor(msim.th),losttriptimes)));
vetot.dl=transpose(msim.dl(ismember(floor(msim.th),losttriptimes)));
vetot.f=transpose(msim.f(ismember(floor(msim.th),losttriptimes)));

% list the injections that were missed outside of these times (ie MISSED in TRIPLE time)
[missed.th,ind]=setdiff(msim.th,vetot.t);
missed.f=msim.f(ind);
missed.d=msim.d(ind);
missed.dh=msim.dh(ind);
missed.dl=msim.dl(ind);
missed.run=msim.run(ind);
missed.hrss_h=msim.hrss_h(ind);
missed.hrss_l=msim.hrss_l(ind);
clear ind 


% list the injections that were FOUND (as DOUBLES) in DOUBLE time

fsimL1H2dd.th=fsimL1H2d.th(ismember(floor(fsimL1H2d.th),vetoedinH1));
fsimL1H2dd.tl=fsimL1H2d.tl(ismember(floor(fsimL1H2d.th),vetoedinH1));
fsimL1H2dd.d=fsimL1H2d.d(ismember(floor(fsimL1H2d.th),vetoedinH1));
fsimL1H2dd.dh=fsimL1H2d.dh(ismember(floor(fsimL1H2d.th),vetoedinH1));
fsimL1H2dd.dl=fsimL1H2d.dl(ismember(floor(fsimL1H2d.th),vetoedinH1));
fsimL1H2dd.f=fsimL1H2d.f(ismember(floor(fsimL1H2d.th),vetoedinH1));
fsimL1H2dd.run=fsimL1H2d.run(ismember(floor(fsimL1H2d.th),vetoedinH1));
fsimL1H2dd.hrss_h=fsimL1H2d.hrss_h(ismember(floor(fsimL1H2d.th),vetoedinH1));
fsimL1H2dd.hrss_l=fsimL1H2d.hrss_l(ismember(floor(fsimL1H2d.th),vetoedinH1));

fsimH1L1dd.th=fsimH1L1d.th(ismember(floor(fsimH1L1d.th),vetoedinH2));
fsimH1L1dd.tl=fsimH1L1d.tl(ismember(floor(fsimH1L1d.th),vetoedinH2));
fsimH1L1dd.d=fsimH1L1d.d(ismember(floor(fsimH1L1d.th),vetoedinH2));
fsimH1L1dd.dh=fsimH1L1d.dh(ismember(floor(fsimH1L1d.th),vetoedinH2));
fsimH1L1dd.dl=fsimH1L1d.dl(ismember(floor(fsimH1L1d.th),vetoedinH2));
fsimH1L1dd.f=fsimH1L1d.f(ismember(floor(fsimH1L1d.th),vetoedinH2));
fsimH1L1dd.run=fsimH1L1d.run(ismember(floor(fsimH1L1d.th),vetoedinH2));
fsimH1L1dd.hrss_h=fsimH1L1d.hrss_h(ismember(floor(fsimH1L1d.th),vetoedinH2));
fsimH1L1dd.hrss_l=fsimH1L1d.hrss_l(ismember(floor(fsimH1L1d.th),vetoedinH2));

fsimH1H2dd.th=fsimH1H2d.th(ismember(floor(fsimH1H2d.th),vetoedinL1));
fsimH1H2dd.tl=fsimH1H2d.tl(ismember(floor(fsimH1H2d.th),vetoedinL1));
fsimH1H2dd.d=fsimH1H2d.d(ismember(floor(fsimH1H2d.th),vetoedinL1));
fsimH1H2dd.dh=fsimH1H2d.dh(ismember(floor(fsimH1H2d.th),vetoedinL1));
fsimH1H2dd.dl=fsimH1H2d.dl(ismember(floor(fsimH1H2d.th),vetoedinL1));
fsimH1H2dd.f=fsimH1H2d.f(ismember(floor(fsimH1H2d.th),vetoedinL1));
fsimH1H2dd.run=fsimH1H2d.run(ismember(floor(fsimH1H2d.th),vetoedinL1));
fsimH1H2dd.hrss_h=fsimH1H2d.hrss_h(ismember(floor(fsimH1H2d.th),vetoedinL1));
fsimH1H2dd.hrss_l=fsimH1H2d.hrss_l(ismember(floor(fsimH1H2d.th),vetoedinL1));

fdoubs=[fsimL1H2dd.th,fsimH1L1dd.th,fsimH1H2dd.th];
doubtimes=[vetoedinH1,vetoedinH2,vetoedinL1];

% list the injections that were  MISSED in DOUBLE time
missedd.th=msim.th(ismember(floor(msim.th),doubtimes));
missedd.tl=msim.tl(ismember(floor(msim.th),doubtimes));
missedd.f=msim.f(ismember(floor(msim.th),doubtimes));
missedd.d=msim.d(ismember(floor(msim.th),doubtimes));
missedd.dh=msim.dh(ismember(floor(msim.th),doubtimes));
missedd.dl=msim.dl(ismember(floor(msim.th),doubtimes));
missedd.run=msim.run(ismember(floor(msim.th),doubtimes));
missedd.hrss_h=msim.hrss_h(ismember(floor(msim.th),doubtimes));
missedd.hrss_l=msim.hrss_l(ismember(floor(msim.th),doubtimes));

% list the injections that were FOUND as DOUBLES in TRIPLE time only
[num,ind]=setdiff(fsimH1L1d.th,fsimH1L1dd.th);
fsimH1L1dt.th=fsimH1L1d.th(ind);
fsimH1L1dt.tl=fsimH1L1d.tl(ind);
fsimH1L1dt.f=fsimH1L1d.f(ind);
fsimH1L1dt.d=fsimH1L1d.d(ind);
fsimH1L1dt.dh=fsimH1L1d.dh(ind);
fsimH1L1dt.dl=fsimH1L1d.dl(ind);
fsimH1L1dt.run=fsimH1L1d.run(ind);
fsimH1L1dt.hrss_h=fsimH1L1d.hrss_h(ind);
fsimH1L1dt.hrss_l=fsimH1L1d.hrss_l(ind);
clear num ind

[num,ind]=setdiff(fsimH1H2d.th,fsimH1H2dd.th);
fsimH1H2dt.th=fsimH1H2d.th(ind);
fsimH1H2dt.tl=fsimH1H2d.tl(ind);
fsimH1H2dt.f=fsimH1H2d.f(ind);
fsimH1H2dt.d=fsimH1H2d.d(ind);
fsimH1H2dt.dh=fsimH1H2d.dh(ind);
fsimH1H2dt.dl=fsimH1H2d.dl(ind);
fsimH1H2dt.run=fsimH1H2d.run(ind);
fsimH1H2dt.hrss_h=fsimH1H2d.hrss_h(ind);
fsimH1H2dt.hrss_l=fsimH1H2d.hrss_l(ind);
clear num ind

[num,ind]=setdiff(fsimL1H2d.th,fsimL1H2dd.th);
fsimL1H2dt.th=fsimL1H2d.th(ind);
fsimL1H2dt.tl=fsimL1H2d.tl(ind);
fsimL1H2dt.f=fsimL1H2d.f(ind);
fsimL1H2dt.d=fsimL1H2d.d(ind);
fsimL1H2dt.dh=fsimL1H2d.dh(ind);
fsimL1H2dt.dl=fsimL1H2d.dl(ind);
fsimL1H2dt.run=fsimL1H2d.run(ind);
fsimL1H2dt.hrss_h=fsimL1H2d.hrss_h(ind);
fsimL1H2dt.hrss_l=fsimL1H2d.hrss_l(ind);

%totals
doubsintrip.th=[fsimH1L1dt.th,fsimH1H2dt.th,fsimL1H2dt.th];

% Bookkeeping
% The number of missed and found injections in each category should equal the total number 
% of missed and found injections read in
total=length(msim.th(ismember(floor(msim.th),vetoedinH1H2L1)))... % missed because of H1H2L1 veto
     +length(msim.th(ismember(floor(msim.th),vetoedinH1H2)))...   % missed because of H1H2 veto
     +length(msim.th(ismember(floor(msim.th),vetoedinH1L1)))...   % missed because of H1L1 veto
     +length(msim.th(ismember(floor(msim.th),vetoedinL1H2)))...   % missed because of H2L1 veto
     +length(msim.th(ismember(floor(msim.th),vetoedinH1)))...     % missed in double time
     +length(msim.th(ismember(floor(msim.th),vetoedinH2)))...     % missed in double time
     +length(msim.th(ismember(floor(msim.th),vetoedinL1)))...     % missed in double time (missedd.th etc)
     +length(missed.f)...                                       % missed in triple time
     +length(injH1t.t)...                                       % found in triple time
     +length(doubsintrip.th)...                                  % found doubles in triple time
     +length(fdoubs);                                         % found doubles in double time

readin=length(fsim.th)-length(injH1inL1dcut.ind )-length(injH2inL1dcut.ind )-length(injH1inH2dcut.ind )+length(msim.th);

%%%%%%%%%%%%%%%%%% PLOTS %%%%%%%%%%%%%%%%%%%%%%%%%%%\

% Hanford effectve distance vs frequency, all missed/found
figure(1)
if exist('injH1t','var')
  ht=loglog(fsim.f(injH1t.ind),fsim.dh(injH1t.ind),'x');
  hold on
end
hH1H2=loglog([fsimH1H2dt.f,fsimH1H2dd.f],[fsimH1H2dt.dh,fsimH1H2dd.dh],'g*');
hold on
hH1L1=loglog([fsimH1L1dt.f,fsimH1L1dd.f],[fsimH1L1dt.dh,fsimH1L1dd.dh],'c*');
hL1H2=loglog([fsimL1H2dt.f,fsimL1H2dd.f],[fsimL1H2dt.dh,fsimL1H2dd.dh],'m*');
if length(veto_list)>0
  hv=semilogy([fsimH1H2dd.f,fsimH1L1dd.f,fsimL1H2dd.f,vetot.f],[fsimH1H2dd.dh,fsimH1L1dd.dh,fsimL1H2dd.dh,vetot.dh],'ko');
end
%hm=loglog([missed.f;missedd.f],[missed.dh;missedd.dh],'r.');
hm=loglog(msim.f,msim.dh,'r.');
%hl=plot([50,50],[1e-2,1e5],'k');
%hu=plot([2000,2000],[1e-2,1e5],'k');
grid on
grid minor
h_xlab=xlabel('f / Hz');
h_ylab=ylabel('d_H / Mpc');
if length(veto_list)>0
  hleg=legend('triples','H1H2 doubles','H1L1 doubles','L1H2 doubles','vetoed','missed');
else
  hleg=legend('triples','H1H2 doubles','H1L1 doubles','L1H2 doubles','missed');
end
eval(['title(''' veto_type ' ' inj_type ': Missed/found plot of frequency vs Hanford d_{eff}'')']);
set(h_xlab,'FontSize',16,'FontName','Times');
set(h_ylab,'FontSize',16,'FontName','Times');
set(gca,'FontSize',16,'FontName','Times');
eval(['saveas(gcf,''' veto_type '_' inj_type '_MF_allH.png'')'])

% Livingston effective distance v's frequency, all missed/found
figure(2)
if exist('injH1t','var')
  ht=loglog(fsim.f(injH1t.ind),fsim.dl(injH1t.ind),'x');
  hold on
end
hH1H2=loglog([fsimH1H2dt.f,fsimH1H2dd.f],[fsimH1H2dt.dl,fsimH1H2dd.dl],'g*');
hold on
hH1L1=loglog([fsimH1L1dt.f,fsimH1L1dd.f],[fsimH1L1dt.dl,fsimH1L1dd.dl],'c*');
hL1H2=loglog([fsimL1H2dt.f,fsimL1H2dd.f],[fsimL1H2dt.dl,fsimL1H2dd.dl],'m*');
if length(veto_list)>0
  hv=semilogy([fsimH1H2dd.f,fsimH1L1dd.f,fsimL1H2dd.f,vetot.f],[fsimH1H2dd.dl,fsimH1L1dd.dl,fsimL1H2dd.dl,vetot.dl],'ko');
end
%hm=loglog([missed.f;missedd.f],[missed.dl;missedd.dl],'ro');
hm=loglog(msim.f,msim.dl,'r.');
%hl=plot([50,50],[1e-2,1e5],'k');
%hu=plot([2000,2000],[1e-2,1e5],'k');
grid on
grid minor
h_xlab=xlabel('f / Hz');
h_ylab=ylabel('d_L / Mpc');
if length(veto_list)>0
  hleg=legend('triples','H1H2 doubles','H1L1 doubles','L1H2 doubles','vetoed','missed');
else
  hleg=legend('triples','H1H2 doubles','H1L1 doubles','L1H2 doubles','missed');
end
eval(['title(''' veto_type ' ' inj_type ': Missed/found plot of frequency vs Livingston d_{eff}'')']);
set(h_xlab,'FontSize',16,'FontName','Times');
set(h_ylab,'FontSize',16,'FontName','Times');
set(gca,'FontSize',16,'FontName','Times');
eval(['saveas(gcf,''' veto_type '_' inj_type '_MF_allL.png'')'])

% Hanford effective distance versus frequency, missed injs
figure(3)
%hm=loglog([missed.f;missedd.f],[missed.dh;missedd.dh],'ro');
hm=loglog(msim.f,msim.dh,'r.');
hold on
if length(veto_list)>0
  hv=semilogy(vetot.f,vetot.dh,'ko');
end
%hl=plot([50,50],[1e-2,1e5],'k');
%hu=plot([2000,2000],[1e-2,1e5],'k');
grid on
grid minor
h_xlab=xlabel('f / Hz');
h_ylab=ylabel('d_H / Mpc');
if length(veto_list)>0
  hleg=legend('missed','vetoed');
else
  hleg=legend('missed');
end
eval(['title(''' veto_type ' ' inj_type ': Missed injections, frequency vs Hanford d_{eff}'')']);
set(h_xlab,'FontSize',16,'FontName','Times');
set(h_ylab,'FontSize',16,'FontName','Times');
set(gca,'FontSize',16,'FontName','Times');
eval(['saveas(gcf,''' veto_type '_' inj_type '_M_H.png'')'])

% Livingston effective distance versus frequency, missed injs
figure(4)
%hm=loglog([missed.f;missedd.f],[missed.dl;missedd.dl],'ro');
hm=loglog(msim.f,msim.dl,'r.');
hold on
if length(veto_list)>0
  hv=semilogy(vetot.f,vetot.dl,'ko');
end
%hl=plot([50,50],[1e-2,1e5],'k');
%hu=plot([2000,2000],[1e-2,1e5],'k');
grid on
grid minor
h_xlab=xlabel('f / Hz');
h_ylab=ylabel('d_L / Mpc');
if length(veto_list)>0
  hleg=legend('missed','vetoed');
else
  hleg=legend('missed');
end
eval(['title(''' veto_type ' ' inj_type ': Missed injections, frequency vs Livingston d_{eff}'')']);
set(h_xlab,'FontSize',16,'FontName','Times');
set(h_ylab,'FontSize',16,'FontName','Times');
set(gca,'FontSize',16,'FontName','Times');
eval(['saveas(gcf,''' veto_type '_' inj_type '_M_L.png'')'])


% Hanford effectve distance vs frequency, doubles
figure(5)
hH1H2=loglog([fsimH1H2dt.f,fsimH1H2dd.f],[fsimH1H2dt.dh,fsimH1H2dd.dh],'g*');
hold on
hH1L1=loglog([fsimH1L1dt.f,fsimH1L1dd.f],[fsimH1L1dt.dh,fsimH1L1dd.dh],'c*');
hL1H2=loglog([fsimL1H2dt.f,fsimL1H2dd.f],[fsimL1H2dt.dh,fsimL1H2dd.dh],'m*');
if length(veto_list)>0
  hv=semilogy([fsimH1H2dd.f,fsimH1L1dd.f,fsimL1H2dd.f],[fsimH1H2dd.dh,fsimH1L1dd.dh,fsimL1H2dd.dh],'ko');
end
%hl=plot([50,50],[1e-2,1e5],'k');
%hu=plot([2000,2000],[1e-2,1e5],'k');
grid on
grid minor
h_xlab=xlabel('f / Hz');
h_ylab=ylabel('d_H / Mpc');
if length(veto_list)>0
  hleg=legend('H1H2 doubles','H1L1 doubles','L1H2 doubles','vetoed');
else
  hleg=legend('H1H2 doubles','H1L1 doubles','L1H2 doubles');
end
eval(['title(''' veto_type ' ' inj_type ': Found doubles, frequency vs Hanford d_{eff}'')']);
set(h_xlab,'FontSize',16,'FontName','Times');
set(h_ylab,'FontSize',16,'FontName','Times');
set(gca,'FontSize',16,'FontName','Times');
eval(['saveas(gcf,''' veto_type '_' inj_type '_F_dH.png'')'])

% Livingston effectve distance vs frequency, doubles
figure(6)
hH1H2=loglog([fsimH1H2dt.f,fsimH1H2dd.f],[fsimH1H2dt.dl,fsimH1H2dd.dl],'g*');
hold on
hH1L1=loglog([fsimH1L1dt.f,fsimH1L1dd.f],[fsimH1L1dt.dl,fsimH1L1dd.dl],'c*');
hL1H2=loglog([fsimL1H2dt.f,fsimL1H2dd.f],[fsimL1H2dt.dl,fsimL1H2dd.dl],'m*');
if length(veto_list)>0
  hv=semilogy([fsimH1H2dd.f,fsimH1L1dd.f,fsimL1H2dd.f],[fsimH1H2dd.dl,fsimH1L1dd.dl,fsimL1H2dd.dl],'ko');
end
%hl=plot([50,50],[1e-2,1e5],'k');
%hu=plot([2000,2000],[1e-2,1e5],'k');
grid on
grid minor
h_xlab=xlabel('f / Hz');
h_ylab=ylabel('d_L / Mpc');
if length(veto_list)>0
  hleg=legend('H1H2 doubles','H1L1 doubles','L1H2 doubles','vetoed');
else
  hleg=legend('H1H2 doubles','H1L1 doubles','L1H2 doubles');
end
eval(['title(''' veto_type ' ' inj_type ': Found doubles, frequency vs Livingston d_{eff}'')']);
set(h_xlab,'FontSize',16,'FontName','Times');
set(h_ylab,'FontSize',16,'FontName','Times');
set(gca,'FontSize',16,'FontName','Times');
eval(['saveas(gcf,''' veto_type '_' inj_type '_F_dL.png'')'])


%%%%%%%%%%%%%%%%%%%%%%% EFFICIENCY %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = 2.303;
rhoBL = 0.0198;
flow = 50;
fhigh = 2000;
nbins = 30;
first=10^(-2);
last=10^2;

% make bins for histogramming
binw = (log10(last)-log10(first))/nbins;    % bin width
% left edge of each bin. the last entry is the far right edge of range
binleft = (log10(first)+binw*(0:(nbins)))'; 
% the center of the bins
bincent = 10.^(binleft+binw/2);  
% get rid of last entry since this isn't a bin (its just a boundary)
bincent = bincent(1:end-1);             

% get indices of injections that were found as triples
tripind=intersect(injH1t.ind,fsim.ind(fsim.f>flow&fsim.f<fhigh));
% bin the found injections
Nfound=histc(log10(fsim.d(tripind)),binleft);
% get rid of the last bin as its not really a bin 
% (this is necessary to do as histc doesnt count anything beyound the last entry 
% in binleft. Thus if the left edge of the last bi was the last entry in bin left
% none of these injections would be counted.

Nfound=Nfound(1:end-1);% missed +  found = total number of injections made
total=[fsim.d(fsim.f>flow&fsim.f<fhigh);msim.d(msim.f>flow&msim.f<fhigh)];
Ntotal=histc(log10(total),binleft); % bin the injections
Ntotal=Ntotal(1:end-1);


% calculate the efficiency and binomial error bars
eff=Nfound./Ntotal;
err=sqrt(eff.*(1-eff)./Ntotal);

tintt.dV=4.*pi.*log(10).*binw.*bincent.^3 ;
tintt.V=sum(tintt.dV.*eff)

%{
% include errors 
sigV=sqrt(sum(err.^2.*tintt.dV.^2));
cal=tintt.V*(1-1./(1+0.05)^3);
toterr=sqrt(sigV^2+cal^2);
err90pc=toterr*1.64;
%tintt.V=tintt.V-err90pc;   % uncomment to include errors
tintt.r=(tintt.V/pi*3/4)^(1/3);         % "effective reach"
tintt.T = 0.0375;
tintt.VT = tintt.V.*tintt.T;
tintt.R = N / rhoBL ./ tintt.T ./ tintt.V;
%}

%plot efficiency curve
figure
h_p1=semilogx(bincent,eff,'bo');
set(h_p1,'MarkerSize',7);
hold on
h_p2=semilogx(bincent,eff);
set(h_p2,'LineWidth',2)
for n=1:length(eff)    % plot the error bars
  h_pe=semilogx([bincent(n) bincent(n)],[eff(n)-err(n) eff(n)+err(n)],'r.-');
  set(h_pe,'MarkerSize',15,'LineWidth',2)
end
hold off
h_xlab=xlabel('d / Mpc');
set(h_xlab,'FontSize',16,'FontName','Times');
h_ylab=ylabel('\epsilon');
set(h_ylab,'FontSize',16,'FontName','Times');
eval(['title(''' veto_type ' ' inj_type ' Efficiency vs distance for triples in f band ' num2str(flow) '-' num2str(fhigh) 'Hz'')']);
grid on
set(gca,'FontSize',16,'FontName','Times');
axis([1e-3 1e3 0 1])
eval(['saveas(gcf,''' veto_type '_' inj_type '_effvd_trip.png'')'])

