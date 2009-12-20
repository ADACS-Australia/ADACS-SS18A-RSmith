function separate(trig_type, inj_type, veto_type, file_list )

%
% NULL = separate ( trig_type, inj_type, veto_type, file_list )
%
% separate breaks trigger files into subfiles (stored as .mat binaries) for each type of trigger: H1H2 doubles,
% H1H2L1 triples, etc.
%
% trig_type = 'bg', 'inj', 'pg', 'int'
%     describes the type of run: background (timeslides), injections, playground, intime (zero lag) respectively
%
% inj_type = 'EOBNR', 'PHENOM', 'RINGDOWN', 'ALL', for injections,
%             0 for background, playground, intime
% veto_type= 'NOVETO', 'CAT2', 'CAT23'
%
% file_list
%     is a column vector of filenames, typically output of lalapps_coincringread that you wish to separate.
%
% EXAMPLE:
%
% file_list = {'injH1H2L1coincs_m1-3.xml'; 'injH1H2L1coincs_m4-6.xml'};
% note that curly brackets {} are necessary, as otherwise file names with varying lengths cannot be read in
% separate( 'inj', 'EOBNR','CAT2', file_list );
%

%N_files = length(file_list(:,1));
N_files = length(file_list);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% read in the file(s) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% read injection files
  if strcmp(trig_type,'inj')
    %create the structure by reading in the first file
%    eval(['coincs=readMeta(file_list{1,:},''sngl_ringdown'',0,''ifo,start_time,start_time_ns,frequency,quality,mass,spin,epsilon,eff_dist,snr,ds2_H1L1,ds2_H2L1,ds2_H1H2,event_id'');'])
   eval(['coincs=readMeta(file_list{1},''sngl_ringdown'',0,''ifo,start_time,start_time_ns,frequency,quality,mass,spin,epsilon,eff_dist,snr,ds2_H1L1,ds2_H2L1,ds2_H1H2,event_id'');'])

    for k=1:length(coincs.snr)
      coincs.run(k)=1;  % this is just an index to identify the injection run
    end
    coincs.run=transpose(coincs.run);

    % read in the rest of the injection files
    for i=2:N_files
      eval(['coincsi=readMeta( file_list{i},''sngl_ringdown'',0,''ifo,start_time,start_time_ns,frequency,quality,mass,spin,epsilon,eff_dist,snr,ds2_H1L1,ds2_H2L1,ds2_H1H2,event_id'');'])
      for k=1:length(coincsi.snr)
        coincsi.run(k)=i;
      end
      coincsi.run=transpose(coincsi.run);
      coincs.ifo=[coincs.ifo;coincsi.ifo];
      coincs.start_time=[coincs.start_time;coincsi.start_time];
      coincs.start_time_ns=[coincs.start_time_ns;coincsi.start_time_ns];
      coincs.frequency=[coincs.frequency;coincsi.frequency];
      coincs.quality=[coincs.quality;coincsi.quality];
      coincs.mass=[coincs.mass;coincsi.mass];
      coincs.spin=[coincs.spin;coincsi.spin];
      coincs.eff_dist=[coincs.eff_dist;coincsi.eff_dist];
      coincs.snr=[coincs.snr;coincsi.snr];
      coincs.ds2_h1l1=[coincs.ds2_h1l1;coincsi.ds2_h1l1];
      coincs.ds2_h1h2=[coincs.ds2_h1h2;coincsi.ds2_h1h2];
      coincs.ds2_h2l1=[coincs.ds2_h2l1;coincsi.ds2_h2l1];
      coincs.event_id=[coincs.event_id;coincsi.event_id];
      coincs.run=[coincs.run;coincsi.run];
      coincs.epsilon=[coincs.epsilon;coincsi.epsilon];
    end
  end

% read background, playground or intime
  if strcmp(trig_type,'bg')||strcmp(trig_type,'pg')||strcmp(trig_type,'int')

    coincs=readMeta( file_list(1,:),'sngl_ringdown',0,'ifo,start_time,start_time_ns,frequency,quality,mass,spin,epsilon,eff_dist,snr,ds2_H1L1,ds2_H2L1,ds2_H1H2,event_id');
    % add a field which says which run a trigger is from
    for k=1:length(coincs.snr)
      coincs.run(k)=1;
    end 
  end

    % simplify the names
    coincs.run=transpose(coincs.run);
    coincs.t=coincs.start_time+coincs.start_time_ns/1e9;
    coincs.f=coincs.frequency;
    coincs.q=coincs.quality;
    coincs.m=coincs.mass;
    coincs.a=coincs.spin;
    coincs.d=coincs.eff_dist;
    coincs.id=coincs.event_id;
    coincs.dst=coincs.epsilon;


%%%%%%%%%%%%%%%%%%%%%%%%% separate into doubles and triples %%%%%%%%%%%%%%%%%%%%%%%%%%


% create an index which is the same for each member of a triple or double
  j=1;
  coincs.ind(1)=1;
  for i=2:length(coincs.f)
    if strcmp(coincs.id(i),coincs.id(i-1))
      coincs.ind(i)=j;
    else
      j=j+1;
      coincs.ind(i)=j;
    end
  end
 

% spilt the triggers into double and triples
i=1;
j=1;
k=1;
triple=0;
double=0;

while i<=length(coincs.ifo)-2
  if isequal(coincs.id(i),coincs.id(i+1),coincs.id(i+2))
     trip.t(j:j+2)=coincs.t(i:i+2);
     trip.f(j:j+2)=coincs.f(i:i+2);
     trip.q(j:j+2)=coincs.q(i:i+2);
     trip.m(j:j+2)=coincs.m(i:i+2);
     trip.a(j:j+2)=coincs.a(i:i+2);
     trip.ifo(j:j+2)=coincs.ifo(i:i+2);
     trip.snr(j:j+2)=coincs.snr(i:i+2);
     trip.ds2_h1l1(j:j+2)=coincs.ds2_h1l1(i:i+2);
     trip.ds2_h1h2(j:j+2)=coincs.ds2_h1h2(i:i+2);
     trip.ds2_h2l1(j:j+2)=coincs.ds2_h2l1(i:i+2);
     trip.id(j:j+2)=coincs.id(i:i+2);
     trip.d(j:j+2)=coincs.d(i:i+2);
     trip.ind(j:j+2)=coincs.ind(i:i+2);
     trip.run(j:j+2)=coincs.run(i:i+2);
     trip.dst(j:j+2)=coincs.dst(i:i+2);
     j=j+3;
     i=i+3;
     triple=triple+1;
  elseif isequal(isequal(coincs.id(i),coincs.id(i+1)),~isequal(coincs.id(i+1),coincs.id(i+2)))
     doub.t(k:k+1)=coincs.t(i:i+1);
     doub.f(k:k+1)=coincs.f(i:i+1);
     doub.q(k:k+1)=coincs.q(i:i+1);
     doub.m(k:k+1)=coincs.m(i:i+1);
     doub.a(k:k+1)=coincs.a(i:i+1);
     doub.ifo(k:k+1)=coincs.ifo(i:i+1);
     doub.snr(k:k+1)=coincs.snr(i:i+1);
     doub.ds2_h1l1(k:k+1)=coincs.ds2_h1l1(i:i+1);
     doub.ds2_h2l1(k:k+1)=coincs.ds2_h2l1(i:i+1);
     doub.ds2_h1h2(k:k+1)=coincs.ds2_h1h2(i:i+1);
     doub.id(k:k+1)=coincs.id(i:i+1);
     doub.d(k:k+1)=coincs.d(i:i+1);
     doub.ind(k:k+1)=coincs.ind(i:i+1);
     doub.run(k:k+1)=coincs.run(i:i+1);
     doub.dst(k:k+1)=coincs.dst(i:i+1);
     k=k+2;
     i=i+2;
     double=double+1;
  else
     error('mmm... something isnt quite right')
  end
end

if ~isequal(coincs.ind(length(coincs.ind)-2),coincs.ind(length(coincs.ind)))&...
    isequal(coincs.ind(length(coincs.ind)-1),coincs.ind(length(coincs.ind)))
  doub.t(k:k+1)=coincs.t(i:i+1);
  doub.f(k:k+1)=coincs.f(i:i+1);
  doub.q(k:k+1)=coincs.q(i:i+1);
  doub.m(k:k+1)=coincs.m(i:i+1);
  doub.a(k:k+1)=coincs.a(i:i+1);
  doub.ifo(k:k+1)=coincs.ifo(i:i+1);
  doub.snr(k:k+1)=coincs.snr(i:i+1);
  doub.ds2_h1l1(k:k+1)=coincs.ds2_h1l1(i:i+1);
  doub.ds2_h2l1(k:k+1)=coincs.ds2_h2l1(i:i+1);
  doub.ds2_h1h2(k:k+1)=coincs.ds2_h1h2(i:i+1);
  doub.id(k:k+1)=coincs.id(i:i+1);
  doub.d(k:k+1)=coincs.d(i:i+1);
  doub.ind(k:k+1)=coincs.ind(i:i+1);
  doub.run(k:k+1)=coincs.run(i:i+1);
  doub.dst(k:k+1)=coincs.dst(i:i+1);
end

%separate triples by ifo
i=1;
x=1;
y=1;
z=1;
if triple>0
  while i<=length(trip.ifo)
    if strcmp(trip.ifo(i),'H1')
      trigH1t.t(x)=trip.t(i);
      trigH1t.f(x)=trip.f(i);
      trigH1t.q(x)=trip.q(i);
      trigH1t.m(x)=trip.m(i);
      trigH1t.a(x)=trip.a(i);
      trigH1t.snr(x)=trip.snr(i);
      trigH1t.ds2_h1l1(x)=trip.ds2_h1l1(i);
      trigH1t.ds2_h1h2(x)=trip.ds2_h1h2(i);
      trigH1t.ds2_h2l1(x)=trip.ds2_h2l1(i);
      trigH1t.id(x)=trip.id(i);
      trigH1t.d(x)=trip.d(i);
      trigH1t.ind(x)=trip.ind(i);
      trigH1t.run(x)=trip.run(i);
      trigH1t.dst(x)=trip.dst(i);
      x=x+1;
    elseif strcmp(trip.ifo(i),'H2')
      trigH2t.t(y)=trip.t(i);
      trigH2t.f(y)=trip.f(i);
      trigH2t.q(y)=trip.q(i);
      trigH2t.m(y)=trip.m(i);
      trigH2t.a(y)=trip.a(i);
      trigH2t.snr(y)=trip.snr(i);
      trigH2t.ds2_h1l1(y)=trip.ds2_h1l1(i);
      trigH2t.ds2_h1h2(y)=trip.ds2_h1h2(i);
      trigH2t.ds2_h2l1(y)=trip.ds2_h2l1(i);
      trigH2t.id(y)=trip.id(i);
      trigH2t.d(y)=trip.d(i);
      trigH2t.ind(y)=trip.ind(i);
      trigH2t.run(y)=trip.run(i);
      trigH2t.dst(y)=trip.dst(i);
      y=y+1;
    elseif strcmp(trip.ifo(i),'L1')
      trigL1t.t(z)=trip.t(i);
      trigL1t.f(z)=trip.f(i);
      trigL1t.q(z)=trip.q(i);
      trigL1t.m(z)=trip.m(i);
      trigL1t.a(z)=trip.a(i);
      trigL1t.snr(z)=trip.snr(i);
      trigL1t.ds2_h1l1(z)=trip.ds2_h1l1(i);
      trigL1t.ds2_h1h2(z)=trip.ds2_h1h2(i);
      trigL1t.ds2_h2l1(z)=trip.ds2_h2l1(i);
      trigL1t.id(z)=trip.id(i);
      trigL1t.d(z)=trip.d(i);
      trigL1t.ind(z)=trip.ind(i);
      trigL1t.run(z)=trip.run(i);
      trigL1t.dst(z)=trip.dst(i);
      z=z+1;
    end
    i=i+1;
  end
end

% save as a mat file
if triple
  if strcmp(trig_type,'inj')
    eval(['save ' veto_type '_' trig_type '_' inj_type '_H1trip.mat -struct trigH1t'])
    eval(['save ' veto_type '_' trig_type '_' inj_type '_H2trip.mat -struct trigH2t'])
    eval(['save ' veto_type '_' trig_type '_' inj_type '_L1trip.mat -struct trigL1t'])
  else
    eval(['save ' veto_type '_' trig_type '_H1_trip.mat -struct trigH1t'])
    eval(['save ' veto_type '_' trig_type '_H2_trip.mat -struct trigH2t'])
    eval(['save ' veto_type '_' trig_type '_L1_trip.mat -struct trigL1t'])
  end
end


%separate doubles by ifo
i=1;
x=1;
y=1;
z=1;
if double>0
  while i<=length(doub.ifo)
    if strcmp(doub.ifo(i),'H1')
      trigH1d.t(x)=doub.t(i);
      trigH1d.f(x)=doub.f(i);
      trigH1d.q(x)=doub.q(i);
      trigH1d.m(x)=doub.m(i);
      trigH1d.a(x)=doub.a(i);
      trigH1d.snr(x)=doub.snr(i);
      trigH1d.ds2_h1l1(x)=doub.ds2_h1l1(i);
      trigH1d.ds2_h2l1(x)=doub.ds2_h2l1(i);
      trigH1d.ds2_h1h2(x)=doub.ds2_h1h2(i);
      trigH1d.id(x)=doub.id(i);
      trigH1d.d(x)=doub.d(i);
      trigH1d.ind(x)=doub.ind(i);
      trigH1d.run(x)=doub.run(i);
      trigH1d.dst(x)=doub.dst(i);
      x=x+1;
    elseif strcmp(doub.ifo(i),'H2')
      trigH2d.t(y)=doub.t(i);
      trigH2d.f(y)=doub.f(i);
      trigH2d.q(y)=doub.q(i);
      trigH2d.m(y)=doub.m(i);
      trigH2d.a(y)=doub.a(i);
      trigH2d.snr(y)=doub.snr(i);
      trigH2d.ds2_h1l1(y)=doub.ds2_h1l1(i);
      trigH2d.ds2_h2l1(y)=doub.ds2_h2l1(i);
      trigH2d.ds2_h1h2(y)=doub.ds2_h1h2(i);
      trigH2d.id(y)=doub.id(i);
      trigH2d.d(y)=doub.d(i);
      trigH2d.ind(y)=doub.ind(i);
      trigH2d.run(y)=doub.run(i);
      trigH2d.dst(y)=doub.dst(i);
      y=y+1;
    elseif strcmp(doub.ifo(i),'L1')
      trigL1d.t(z)=doub.t(i);
      trigL1d.f(z)=doub.f(i);
      trigL1d.q(z)=doub.q(i);
      trigL1d.m(z)=doub.m(i);
      trigL1d.a(z)=doub.a(i);
      trigL1d.snr(z)=doub.snr(i);
      trigL1d.ds2_h1l1(z)=doub.ds2_h1l1(i);
      trigL1d.ds2_h2l1(z)=doub.ds2_h2l1(i);
      trigL1d.ds2_h1h2(z)=doub.ds2_h1h2(i);
      trigL1d.id(z)=doub.id(i);
      trigL1d.d(z)=doub.d(i);
      trigL1d.ind(z)=doub.ind(i);
      trigL1d.run(z)=doub.run(i);
      trigL1d.dst(z)=doub.dst(i);
      z=z+1;
    end
    i=i+1;
  end

% put the H1L1 doubles in a structure
  [com,H1,L1]=intersect(trigH1d.ind,trigL1d.ind);
  trigH1inL1d.ind=trigH1d.ind(H1);
  trigH1inL1d.t=trigH1d.t(H1);
  trigH1inL1d.f=trigH1d.f(H1);
  trigH1inL1d.q=trigH1d.q(H1);
  trigH1inL1d.m=trigH1d.m(H1);
  trigH1inL1d.a=trigH1d.a(H1);
  trigH1inL1d.id=trigH1d.id(H1);
  trigH1inL1d.d=trigH1d.d(H1);
  trigH1inL1d.snr=trigH1d.snr(H1);
  trigH1inL1d.ds2_h1l1=trigH1d.ds2_h1l1(H1);
  trigH1inL1d.ds2_h1h2=trigH1d.ds2_h1h2(H1);
  trigH1inL1d.ds2_h2l1=trigH1d.ds2_h2l1(H1);
  trigH1inL1d.run=trigH1d.run(H1);
  trigH1inL1d.dst=trigH1d.dst(H1);

  if strcmp(trig_type,'inj')
    eval(['save ' veto_type '_' trig_type '_' inj_type '_H1inL1doub.mat -struct trigH1inL1d'])
  else
    eval(['save ' veto_type '_' trig_type '_H1inL1doub.mat -struct trigH1inL1d'])
  end

  trigL1inH1d.ind=trigL1d.ind(L1);
  trigL1inH1d.t=trigL1d.t(L1);
  trigL1inH1d.f=trigL1d.f(L1);
  trigL1inH1d.q=trigL1d.q(L1);
  trigL1inH1d.m=trigL1d.m(L1);
  trigL1inH1d.a=trigL1d.a(L1);
  trigL1inH1d.id=trigL1d.id(L1);
  trigL1inH1d.d=trigL1d.d(L1);
  trigL1inH1d.snr=trigL1d.snr(L1);
  trigL1inH1d.ds2_h1l1=trigL1d.ds2_h1l1(L1);
  trigL1inH1d.ds2_h1h2=trigL1d.ds2_h1h2(L1);
  trigL1inH1d.ds2_h2l1=trigL1d.ds2_h2l1(L1);
  trigL1inH1d.run=trigL1d.run(L1);
  trigL1inH1d.dst=trigL1d.dst(L1);

  if strcmp(trig_type,'inj')
    eval(['save ' veto_type '_' trig_type '_' inj_type '_L1inH1doub.mat -struct trigL1inH1d'])
  else
    eval(['save ' veto_type '_' trig_type '_L1inH1doub.mat -struct trigL1inH1d'])
  end

% put the H1H2 doubles in a structure
  [com,H1,H2]=intersect(trigH1d.ind,trigH2d.ind);
  trigH1inH2d.ind=trigH1d.ind(H1);
  trigH1inH2d.t=trigH1d.t(H1);
  trigH1inH2d.f=trigH1d.f(H1);
  trigH1inH2d.q=trigH1d.q(H1);
  trigH1inH2d.m=trigH1d.m(H1);
  trigH1inH2d.a=trigH1d.a(H1);
  trigH1inH2d.id=trigH1d.id(H1);
  trigH1inH2d.d=trigH1d.d(H1);
  trigH1inH2d.snr=trigH1d.snr(H1);
  trigH1inH2d.ds2_h1l1=trigH1d.ds2_h1l1(H1);
  trigH1inH2d.ds2_h1h2=trigH1d.ds2_h1h2(H1);
  trigH1inH2d.ds2_h2l1=trigH1d.ds2_h2l1(H1);
  trigH1inH2d.run=trigH1d.run(H1);
  trigH1inH2d.dst=trigH1d.dst(H1);

  if strcmp(trig_type,'inj')
    eval(['save ' veto_type '_' trig_type '_' inj_type '_H1inH2doub.mat -struct trigH1inH2d'])
  else
    eval(['save ' veto_type '_' trig_type '_H1inH2doub.mat -struct trigH1inH2d'])
  end

  trigH2inH1d.ind=trigH2d.ind(H2);
  trigH2inH1d.t=trigH2d.t(H2);
  trigH2inH1d.f=trigH2d.f(H2);
  trigH2inH1d.q=trigH2d.q(H2);
  trigH2inH1d.m=trigH2d.m(H2);
  trigH2inH1d.a=trigH2d.a(H2);
  trigH2inH1d.id=trigH2d.id(H2);
  trigH2inH1d.d=trigH2d.d(H2);
  trigH2inH1d.snr=trigH2d.snr(H2);
  trigH2inH1d.ds2_h1l1=trigH2d.ds2_h1l1(H2);
  trigH2inH1d.ds2_h1h2=trigH2d.ds2_h1h2(H2);
  trigH2inH1d.ds2_h2l1=trigH2d.ds2_h2l1(H2);
  trigH2inH1d.run=trigH2d.run(H2);
  trigH2inH1d.dst=trigH2d.dst(H2);

  if strcmp(trig_type,'inj')
    eval(['save ' veto_type '_' trig_type '_' inj_type '_H2inH1doub.mat -struct trigH2inH1d'])
  else
    eval(['save ' veto_type '_' trig_type '_H2inH1doub.mat -struct trigH2inH1d'])
  end

% put the L1H2 doubles in a structure
  [com,L1,H2]=intersect(trigL1d.ind,trigH2d.ind);
  trigL1inH2d.ind=trigL1d.ind(L1);
  trigL1inH2d.t=trigL1d.t(L1);
  trigL1inH2d.f=trigL1d.f(L1);
  trigL1inH2d.q=trigL1d.q(L1);
  trigL1inH2d.m=trigL1d.m(L1);
  trigL1inH2d.a=trigL1d.a(L1);
  trigL1inH2d.id=trigL1d.id(L1);
  trigL1inH2d.d=trigL1d.d(L1);
  trigL1inH2d.snr=trigL1d.snr(L1);
  trigL1inH2d.ds2_h1l1=trigL1d.ds2_h1l1(L1);
  trigL1inH2d.ds2_h2l1=trigL1d.ds2_h2l1(L1);
  trigL1inH2d.ds2_h1h2=trigL1d.ds2_h1h2(L1);
  trigL1inH2d.run=trigL1d.run(L1);
  trigL1inH2d.dst=trigL1d.dst(L1);

  if strcmp(trig_type,'inj')
    eval(['save ' veto_type '_' trig_type '_' inj_type '_L1inH2doub.mat -struct trigL1inH2d'])
  else
    eval(['save ' veto_type '_' trig_type '_L1inH2doub.mat -struct trigL1inH2d'])
  end

  trigH2inL1d.ind=trigH2d.ind(H2);
  trigH2inL1d.t=trigH2d.t(H2);
  trigH2inL1d.f=trigH2d.f(H2);
  trigH2inL1d.q=trigH2d.q(H2);
  trigH2inL1d.m=trigH2d.m(H2);
  trigH2inL1d.a=trigH2d.a(H2);
  trigH2inL1d.id=trigH2d.id(H2);
  trigH2inL1d.d=trigH2d.d(H2);
  trigH2inL1d.snr=trigH2d.snr(H2);
  trigH2inL1d.ds2_h1l1=trigH2d.ds2_h1l1(H2);
  trigH2inL1d.ds2_h2l1=trigH2d.ds2_h2l1(H2);
  trigH2inL1d.ds2_h1h2=trigH2d.ds2_h1h2(H2);
  trigH2inL1d.run=trigH2d.run(H2);
  trigH2inL1d.dst=trigH2d.dst(H2);

  if strcmp(trig_type,'inj')
    eval(['save ' veto_type '_' trig_type '_' inj_type '_H2inL1doub.mat -struct trigH2inL1d'])
  else
    eval(['save ' veto_type '_' trig_type '_H2inL1doub.mat -struct trigH2inL1d'])
  end
end

