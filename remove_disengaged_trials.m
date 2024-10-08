function [adj_actions, adj_cues, adj_numtrials] = remove_disengaged_trials(day_file, actions, schedule)

% recode mouse's choices into a binary + NaN array
actions_coded = actions;
actions_coded(actions_coded<1.1)=NaN;
actions_coded(actions_coded==2.0200)=NaN;
actions_coded(actions_coded==1.2200)=1; % change to include 1.12
actions_coded(actions_coded==3.0000)=0;

% specify bin size, then find the last trial with hitrate > 50%
binsize = 10;
hitrate=movmean(actions_coded,binsize,'omitnan');
cutoff = find(hitrate >= 0.5, 1, 'last');

%figure;
%plot(actions_coded);hold on;
%plot(hitrate);
%title(day_file);
if cutoff > 1
    adj_actions = actions(1:cutoff);
    adj_cues = schedule(1:cutoff);
    adj_numtrials = cutoff;
else
    adj_actions = [0];
    adj_cues = [0];
    adj_numtrials = 0;
end

