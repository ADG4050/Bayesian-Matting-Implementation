
% This funciton is to calculate Sum of absolute difference 
% between bayesian alpha % and ground truth alpha. 

% note: this function is applicable to version 2018b and above

function sadvalue = sad2d(alpha, GTalpha)
% check size of input before calculate sad

a = sizecheck(alpha, GTalpha);
if a == 1
    diff = abs(alpha - GTalpha);
    sadvalue = sum(diff(:), 'omitnan');
end

end 