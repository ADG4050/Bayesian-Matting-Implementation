% This funciton is to check the size of input is same 
% and  the class is double

% note: this function is applicable to version 2018b and above

function test = sizecheck(alpha, GTalpha)

sz1 = size(alpha);
sz2 = size(GTalpha);

ty1 = class(alpha);
ty2 = class(GTalpha);

a = strcmp (ty1,'double');
b = strcmp (ty2,'double');
  
if a && b
    if sz1 == sz2
        test = 1;
    else
        test = 0;
        fprintf('Matrix dimensions must be consistent');
    end
else
    test = 0;
    fprintf('Unable to handle non-double type data');
end