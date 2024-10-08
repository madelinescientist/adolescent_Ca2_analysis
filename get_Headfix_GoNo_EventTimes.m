function out=get_Headfix_GoNo_EventTimes(varargin)
% out=get_Headfix_GoNo_EventTimes(exper_file_name,recorded frame number)
% get_Headfix_GoNo_EventTimes is a function that reads the Headfix_sound_GoNG data file
% and output a structure variable "out"
% Required input: exper_file_name or exper_variable
% Optional: recorded frame number. This is for extrapolating frame time
% after the last counted trial
% (1) out.GoNG_EventTimes is in three rows: [eventID; eventTime; trial].
% List of eventID for the first row of out.GoNG_EventTimes
% eventID=1:    2P imaging frame TTL high
% eventID=2:    2P imaging frame TTL low
% eventID=3:    left lick in
% eventID=4:    left lick out
% eventID=44:   Last left lick out
% eventID=5:    right lick 1n
% eventID=6:    right lick out
% eventID=66:   Last right lick out
% eventID=7.01:  new trial, Sound 1 ON
% eventID=7.02:  new trial, Sound 2 ON
% eventID=7.0n:  new trial, Sound n ON
% eventID=7.16:  new trial, Sound 16 ON
% eventID=81.01: Correct No-Go (no-lick), unrewarded outcome
% eventID=81.02: Correct Go (lick), unrewarded outcome
% eventID=81.12: Correct Go (lick), 1 drop rewarded after direct delivery
% eventID=81.22: Correct Go (lick), 2 drops rewarded (valve on)
% eventID=82:02  False Go (lick), white noise on
% eventID=83:    Missed to respond
% eventID=84:    Aborted outcome
% eventID=9.01:  Water Valve on 1 time (1 reward)
% eventID=9.02:  Water Valve on 2 times (2 rewards)
% eventID=9.03:  Water Valve on 3 times (3 rewards)
%
% (2) out.sound_name is the name of the sound referenced in eventID=7.x
% (3) out.sound_freq is the frequency of the sound
%     out.sound_dur is the duration of the sound
%     out.sound_SPL is the amplitude of the sound
% (4) out.schedule is the stimulus schedule for each trial
% (5) out.directdelivery =1 if water in the trial was directly delivered
% (6) out.portside is the port_side schedule (-1:NoGo, 0:probe, 2:Go/left)
% (7) out.result is the result for each trial
% (8) out.run_speed is the rotation speed in two rows: [Time; rps].
% (9) out.frame_time is the SPIRAL MODE frame time of the 2P microscope
%      probably won't work in other scanning mode
% (10) out.recovered_frame_time is attemped recovered missing frame time
%      in some cases when trigger is missing mid-session
% 6/11/2022 Lung-Hao Tai

% above are trial based,
% not time based: stimulus ID, scedule ID, set size?
% frame timestamp
% rotation speed, timestamp

out = [];
efn = 0;
if nargin ==1
    arg = varargin{1};
    if ischar(arg) || iscellstr(arg) || isstring(arg)
        filename=arg;
        full_filename=which(filename);
        if isempty(full_filename)
            full_filename=filename;
        end
        dr=dir(full_filename);
        if ~isempty(dr)
            data=load(full_filename);
        else
            data = [];
        end
    elseif isfield(arg, 'exper')
        data = arg;
    else
        data = [];
    end
elseif nargin ==2
    arg = varargin{1};
    efn = varargin{2};
    if ischar(arg) || iscellstr(arg) || isstring(arg)
        filename=arg;
        full_filename=which(filename);
        if isempty(full_filename)
            full_filename=filename;
        end
        dr=dir(full_filename);
        if ~isempty(dr)
            data=load(full_filename);
        else
            data = [];
        end
    elseif isfield(arg, 'exper')
        data = arg;
    else
        data = [];
    end
else
    disp('Please specify an exper filename in string');
    eval('help get_Headfix_GoNo_EventTimes');
    return
end

GoNG_EventTimes=[];
GoNG_EventTimes_n=0;

if ~isempty(data)
    trial_events=data.exper.rpbox.param.trial_events.value;
    if isfield(data.exper,'headfix_sound_gong')
        CountedTrial=data.exper.headfix_sound_gong.param.countedtrial.value;
        Result=data.exper.headfix_sound_gong.param.result.value(1:CountedTrial);
        portside=data.exper.headfix_sound_gong.param.port_side.value(1:CountedTrial);
        schedule=data.exper.headfix_sound_gong.param.schedule.value(1:CountedTrial);
        directdelivery=data.exper.headfix_sound_gong.param.directdelivery.user(1:CountedTrial);
%         lwatervalvedur=data.exper.headfix_sound_gong.param.lwatervalvedur.value;
%         rwatervalvedur=data.exper.headfix_sound_gong.param.rwatervalvedur.value;
%         boxrig=data.exper.control.param.expid.value;
        protocol='headfix_sound_gong';
        StimParam=data.exper.headfix_sound_gong.param.stimparam.value;
        param_string=data.exper.headfix_sound_gong.param.stimparam.user;
        LeftP  =str2double(StimParam(:,strcmp(param_string,'left reward ratio')));
        RightP =str2double(StimParam(:,strcmp(param_string,'right reward ratio')));
        LeftRewardP=LeftP(schedule);
        RightRewardP=RightP(schedule);
    else
        error('no Headfix_Sound_GoNG session found');
        return;
    end

    % start trials based sorting of events
    for k=1:CountedTrial
        if k==1
            tt1=0; % trial timestamp 1
        else
            tt1=tt2;    % new trial timestamp 1 = old trial timestamp 2
        end
        if ~isempty(data.exper.headfix_sound_gong.param.trial_events.trial{k})
            if ismember(floor(Result(k)*10)/10,[1.2, 1.3]) % two or three drop H2O
                tt2=data.exper.headfix_sound_gong.param.trial_events.trial{k}(end,3);
            else
                tt2=data.exper.headfix_sound_gong.param.trial_events.trial{k}(:,3);
                if length(tt2)>1
                    disp('two or more trial_events');
                    disp(data.exper.headfix_sound_gong.param.trial_events.trial{k}(:,1:2))
                    tt2=tt2(1);
                end
            end
        else
            % try to find missing trial_events
            if Result(k)==4 && k<CountedTrial
                % try skip to next trial if exper is jumping one trial
                disp(['found skipped trial in headfix_sound_gong for trial ' num2str(k) ', in file:' filename]);
            elseif Result(k)==0 && k<CountedTrial
                % check if the Result(k) is 0, check if spurious right
                % port false is triggerrd (then Result(k)=2.01 )
                tt3=data.exper.headfix_sound_gong.param.trial_events.trial{k+1}(end,3);
                current_te=trial_events(trial_events(:,2)>tt1 & trial_events(:,2)<tt3, 2:4);
                Right_port_false_time_idx=find(ismember(current_te(:,2),[14 15])&ismember(current_te(:,3),[5 6]));
                if ~isempty(Right_port_false_time_idx)
                    data.exper.headfix_sound_gong.param.trial_events.trial{k}(1,:)=current_te(Right_port_false_time_idx,[2 3 1]);
                    tt2=current_te(Right_port_false_time_idx,1);
                    data.exper.headfix_sound_gong.param.result.value(k)=2.01;
                    Result(k)=2.01;
                    disp(['found spurious right port false in headfix_sound_gong for trial ' num2str(k) ', in file:' filename]);
                else
                    error(['no trial events in headfix_sound_gong for trial ' num2str(k) ', in file:' filename]);
                end

            end
        end

        % trial_events = (trial, time, state, chan, next state))
        current_te=trial_events(trial_events(:,2)>tt1 & trial_events(:,2)<=tt2, 2:4);
        % SoundOn Time
        new_trial_SoundOn_time=current_te(ismember(current_te(:,2),[1 11 21])&ismember(current_te(:,3),8),1);
        if ~isempty(new_trial_SoundOn_time) && length(new_trial_SoundOn_time)==1
            % all events happened before sound-on is in ITI
            ITI_te=trial_events(trial_events(:,2)>tt1 & trial_events(:,2)<new_trial_SoundOn_time & ismember(trial_events(:,4),[3:6]), 2:4);
            last_poke_out=find(ismember(ITI_te(:,3),[4 6]),1,'last');
            if ~isempty(last_poke_out)
                ITI_te(last_poke_out,3)=ITI_te(last_poke_out,3)*10+ITI_te(last_poke_out,3);
            end
            GoNG_EventTimes(:,GoNG_EventTimes_n+1:GoNG_EventTimes_n+length(ITI_te(:,1)))=[ITI_te(:,3)';ITI_te(:,1)';ones(size(ITI_te(:,1)'))*(k-0.5)];
            GoNG_EventTimes_n=GoNG_EventTimes_n+length(ITI_te(:,1));
            % the one trigger a new trial [event=1.1 , time, trial=k]
            SoundID=str2double(StimParam{schedule(k),5})/100;
            GoNG_EventTimes(:,GoNG_EventTimes_n+1)=[7+SoundID ;new_trial_SoundOn_time;k]; % new trial sound on
            GoNG_EventTimes_n=GoNG_EventTimes_n+1;

            %Done with ITI trial events, now look at trial events in Trial K
            if directdelivery(k)==1
                Tk_te=trial_events(trial_events(:,2)>new_trial_SoundOn_time & trial_events(:,2)<=tt2 & ismember(trial_events(:,4),[3:6]), 2:4);
                Tk_te1=trial_events(trial_events(:,2)>new_trial_SoundOn_time & trial_events(:,2)<=tt2 & (trial_events(:,3)==15 & trial_events(:,4)==8), 2:4);
                Tk_te1(:,3)=9.01; % direct delivery eventID
                Tk_te2=trial_events(trial_events(:,2)>new_trial_SoundOn_time & trial_events(:,2)<=tt2 & (trial_events(:,3)==44 & trial_events(:,4)==8), 2:4);
                Tk_te2(:,3)=9.02;
                Tk_te3=trial_events(trial_events(:,2)>new_trial_SoundOn_time & trial_events(:,2)<=tt2 & (trial_events(:,3)==43 & trial_events(:,4)==8), 2:4);
                Tk_te3(:,3)=9.03;
                Tk_te=[Tk_te;Tk_te1;Tk_te2;Tk_te3];
                [Y I]=sort(Tk_te(:,1),1);
                Tk_te=Tk_te(I,:);
            else
                Tk_te=trial_events(trial_events(:,2)>new_trial_SoundOn_time & trial_events(:,2)<=tt2 & ismember(trial_events(:,4),[3:6]), 2:4);
                Tk_te1=trial_events(trial_events(:,2)>new_trial_SoundOn_time & trial_events(:,2)<=tt2 & (trial_events(:,3)==45 & trial_events(:,4)==8), 2:4);
                Tk_te1(:,3)=9.01;
                Tk_te2=trial_events(trial_events(:,2)>new_trial_SoundOn_time & trial_events(:,2)<=tt2 & (trial_events(:,3)==44 & trial_events(:,4)==8), 2:4);
                Tk_te2(:,3)=9.02;
                Tk_te3=trial_events(trial_events(:,2)>new_trial_SoundOn_time & trial_events(:,2)<=tt2 & (trial_events(:,3)==43 & trial_events(:,4)==8), 2:4);
                Tk_te3(:,3)=9.03;
                Tk_te=[Tk_te;Tk_te1;Tk_te2;Tk_te3];
                [Y I]=sort(Tk_te(:,1),1);
                Tk_te=Tk_te(I,:);
            end
            GoNG_EventTimes(:,GoNG_EventTimes_n+1:GoNG_EventTimes_n+length(Tk_te(:,1)))=[Tk_te(:,3)';Tk_te(:,1)';ones(size(Tk_te(:,1)'))*k];
            GoNG_EventTimes_n=GoNG_EventTimes_n+length(Tk_te(:,1));

            %Now look at outcome (tt2) if not already added [event=70+result, time, trial=k]
            GoNG_EventTimes(:,GoNG_EventTimes_n+1)=[80+Result(k);tt2;k]; %
            GoNG_EventTimes_n=GoNG_EventTimes_n+1;
        elseif Result(k)~=4
            error(['no sound on event found for trial ' num2str(k) ', check for error']);
        end

    end
    out.GoNG_EventTimes=GoNG_EventTimes;
    for i=1:8
        out.sound_name{i}=eval(['data.exper.headfix_sound_gong.param.sound' num2str(i) '.list{data.exper.headfix_sound_gong.param.sound' num2str(i) '.value}']);
    end
    if isfield(data.exper.headfix_sound_gong.param,'sound9')
        for i=9:16
            out.sound_name{i}=eval(['data.exper.headfix_sound_gong.param.sound' num2str(i) '.list{data.exper.headfix_sound_gong.param.sound' num2str(i) '.value}']);
        end
    end
    for i=1:8
        out.sound_freq(i)=eval(['data.exper.headfix_sound_gong.param.tonefreq.value{data.exper.headfix_sound_gong.param.sound' num2str(i) '.value}']);
    end
    if isfield(data.exper.headfix_sound_gong.param,'sound9')
        for i=9:16
            out.sound_freq(i)=eval(['data.exper.headfix_sound_gong.param.tonefreq.value{data.exper.headfix_sound_gong.param.sound' num2str(i) '.value}']);
        end
    end
    for i=1:8
        out.sound_dur(i)=eval(['data.exper.headfix_sound_gong.param.tonedur.value(data.exper.headfix_sound_gong.param.sound' num2str(i) '.value)']);
    end
    if isfield(data.exper.headfix_sound_gong.param,'sound9')
        for i=9:16
            out.sound_dur(i)=eval(['data.exper.headfix_sound_gong.param.tonedur.value(data.exper.headfix_sound_gong.param.sound' num2str(i) '.value)']);
        end
    end
    % Max=70dB SPL use 60dB to correct Neo3 off axis drop beyond 10kHz
    for i=1:8
        out.sound_SPL(i)=eval(['data.exper.headfix_sound_gong.param.tonespl.value(data.exper.headfix_sound_gong.param.sound' num2str(i) '.value)']);
        if out.sound_freq(i)>10900
            out.sound_SPL(i)=55;
        end
    end
    if isfield(data.exper.headfix_sound_gong.param,'sound9')
        for i=9:16
            out.sound_SPL(i)=eval(['data.exper.headfix_sound_gong.param.tonespl.value(data.exper.headfix_sound_gong.param.sound' num2str(i) '.value)']);
            if out.sound_freq(i)>10900
                out.sound_SPL(i)=55;
            end
        end
    end
    out.schedule=schedule;
    out.directdelivery=directdelivery;
    portside(LeftRewardP==-1 & RightRewardP==-1)=-1;
    out.portside=portside;
    out.result=Result;
    if isfield(data.exper.headfix_sound_gong.param,'rptime_rotatoryposition')
        rptime_rotatoryposition=data.exper.headfix_sound_gong.param.rptime_rotatoryposition.value;
        % remove NaN
        rptime_rotatoryposition=rptime_rotatoryposition(:,~isnan(rptime_rotatoryposition(1,:)));
        % remove previous session speed (in early protocol, rptime_rotatoryposition was not reset in new session)
        first_idx=find(diff(rptime_rotatoryposition(1,:))<0,1,'last');
        if ~isempty(first_idx)
            rptime_rotatoryposition=rptime_rotatoryposition(:,first_idx+1:end);
        end
        rptime_rotatoryposition(2,2:end)=(diff(rptime_rotatoryposition(2,:))/360)./diff(rptime_rotatoryposition(1,:)); % 360 tick is one rotation
        rptime_rotatoryposition(2,1)=0;
        out.run_speed=rptime_rotatoryposition;
    end

    % now find galvo signal from 2P microscope
    galvo_signal=data.exper.rpbox.param.trial_events.value(ismember(data.exper.rpbox.param.trial_events.value(:,4),[1 2]),[2 4]);
    figure(29);clf;
    axh=axes;
    if length(galvo_signal)>10
        %turn TTL(center) in/out:[1 2] to on/off:[1 0]
        galvo_signal(:,2)=2-galvo_signal(:,2);
        t=galvo_signal(:,1);
        v=galvo_signal(:,2);
        diff_t=diff(t);
        diff_v=diff(v);
        % check all frame duration less than 200 ms
        diff_t_idx= diff_v==1 & diff_t<0.2;
        % add in a 60ms fast frame time to widen the bin range
        [n,x]=hist(axh,[diff_t(diff_t_idx);0.06]);
        bar(axh,x,n,'hist');
        hold on;
        ax=axis;
        maxn_idx=find(n==max(n));
        diff_t_maxbin_idx=diff_t<(x(maxn_idx)+mean(diff(x))*1) & diff_t>(x(maxn_idx)-mean(diff(x))*.5);
        maxbin_t=t(diff_t_maxbin_idx);
        maxbin_v=v(diff_t_maxbin_idx);
%         length(maxbin_t) % number of total total frames
        gt=galvo_signal(:,1)';
        gv=galvo_signal(:,2)';
        gt(2,1:length(gt)-1)=gt(2:end);
        gt(2,end)=nan;
        gv(2,1:length(gv)-1)=gv(1:end-1);
        gv(2,end)=nan;
        plot(axh,(gt(:)-gt(1))*diff(ax(1:2))/(gt(end-1)-gt(1))+ax(1),gv(:)*diff(ax(3:4))/8+diff(ax(3:4))*6/8);
        plot(axh,(maxbin_t-gt(1))*diff(ax(1:2))/(gt(end-1)-gt(1))+ax(1),maxbin_v*diff(ax(3:4))/8+diff(ax(3:4))*6/8,'r.');
        % find if there is missed trigger/missing frames
        frameinterval=diff(maxbin_t);
        mean_frameinterval=mean(frameinterval(frameinterval<0.2));
        frametimejump_idx=find(frameinterval>(mean_frameinterval*1.5));
        if ~isempty(frametimejump_idx)
            for j=1:length(frametimejump_idx)
                n_missing_frame=round((maxbin_t(frametimejump_idx(j)+1)-maxbin_t(frametimejump_idx(j)))/mean_frameinterval);
                missing_frame_time=linspace(maxbin_t(frametimejump_idx(j)),maxbin_t(frametimejump_idx(j)+1),n_missing_frame+1);
                missing_frame_time=missing_frame_time(~ismember(missing_frame_time,maxbin_t));
                new_frame_time=sort([maxbin_t;missing_frame_time']);
            end
        else
            new_frame_time=maxbin_t;
        end
        % extend frame to unfinished trial with recorded frame number (efn)
        if length(new_frame_time)<efn
           new_frame_time2=interp1(1:length(new_frame_time),new_frame_time,(length(new_frame_time)+1):efn,'linear','extrap'); 
           new_frame_time=[new_frame_time; new_frame_time2'];
        end

        plot(axh,(new_frame_time-gt(1))*diff(ax(1:2))/(gt(end-1)-gt(1))+ax(1),ones(size(new_frame_time))*maxbin_v(end)*diff(ax(3:4))/8+diff(ax(3:4))*5.5/8,'g.');
        last_trial_events=data.exper.rpbox.param.trial_events.value(end,2);
        plot(axh,(last_trial_events-gt(1))*diff(ax(1:2))/(gt(end-1)-gt(1))+ax(1),maxbin_v(end)*diff(ax(3:4))/8+diff(ax(3:4))*5.5/8,'ko');
        axis([ax(1) max([ax(2),(last_trial_events-gt(1))*diff(ax(1:2))/(gt(end-1)-gt(1))+ax(1),(new_frame_time(end)-gt(1))*diff(ax(1:2))/(gt(end-1)-gt(1))+ax(1)]) ax(3:4)]);
        txh=text(ax(1)+diff(ax(1:2))*0.2,ax(3)+diff(ax(3:4))*0.82,['found ' num2str(max(n)) ' frames' sprintf('\n') 'mean frame dur=' num2str(mean(diff_t(diff_t_idx))*1000) 'mSec']);
        set(txh,'color',[1 0 0]);
        txh=text(ax(1)+diff(ax(1:2))*0.2,ax(3)+diff(ax(3:4))*0.66,['potentially ' num2str(length(new_frame_time)) ' frames']);
        set(txh,'color',[0 .6 .1]);
        out.frame_time=maxbin_t;
        out.recovered_frame_time=new_frame_time;
    else
        disp('very few frame from galvo signal');
    end
    xlabel(axh,'Frame duration (Secs)');
    ylabel(axh,'Frames');
    set(axh,'tag','plot_frames');


else
    disp('file not found');
end
