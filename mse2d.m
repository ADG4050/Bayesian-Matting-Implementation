
% This funciton is to calculate Mean squred error between bayesian alpha 
% and ground truth alpha. 

% note: this function is applicable to version 2018b and above

function msevalue = mse2d(alpha, GTalpha)

% check size of input before calculate mse
a = sizecheck(alpha, GTalpha);

if a == 1
    diff = (alpha - GTalpha);
    msevalue = mean(diff(:).^2, 'omitnan');

end

end 

