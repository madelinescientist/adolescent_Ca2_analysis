function [out] = gonogo_extract_behavior_data(folder, animal, session)
    % (folder, animal, session) -> filename of the behavior_raw  fullfile
    %filename = fullfile(folder, 'raw_behavior/to_process', animal, session);
    fname = strcat(animal,'-',session,'-gonogo.mat');
    disp(fname);
    filename = fullfile(folder,'/raw_behavior/to_process/',animal,fname);
    disp(filename);
    output_folder = fullfile(folder, '/processed_data/Data', animal, session);
    if ~exist(output_folder, 'dir')
        mkdir(output_folder);
    end
    behav_log = fullfile(output_folder, [animal, '_', session, '_', 'behaviorLOG.mat']);
    %if ~exist(behav_log, 'file')
        out = get_Headfix_GoNo_EventTimes(filename);
        save(behav_log, '-v7.3', 'out');
    %end
end