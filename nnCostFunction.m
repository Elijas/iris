function [J grad] = nnCostFunction(nn_params, nn_lsizes, X, y, lambda)
    y = eye(nn_lsizes(end))(y,:);

    a = X; %a contains horizontal vectors of neuron activation values;
    Jvec_params = []; %Useful for later regularization of Cost Function
    gradvec_params = []; %Useful for later to regularize gradients
    a_pos = 0;
    params_pos = 0;
    for i=1:size(nn_lsizes,2)-1;
        ThetaRows = nn_lsizes(i+1);
        ThetaCols = nn_lsizes(i)+1;
        Theta = reshape(nn_params(params_pos+1 : params_pos+ThetaRows*ThetaCols), ThetaRows, ThetaCols);

        a = [a  sigmoid([ones(size(a,1),1) a(:,a_pos+1:a_pos+nn_lsizes(i))] * Theta')];

        Jvec_params = [Jvec_params; Theta(:,2:end)(:)];
        Theta(:,1)=0;
        gradvec_params = [gradvec_params; Theta(:)];

        params_pos += ThetaRows*ThetaCols;
        a_pos += nn_lsizes(i);
    end

    % COST
    Jvec_h = a(:,end-nn_lsizes(end)+1:end)(:);
    Jvec_y = y(:);
    J = -1/size(a,1) .* (Jvec_y'*log(Jvec_h) + (1-Jvec_y)'*log(1-Jvec_h));
    J += lambda/(2*size(a,1)) * sum(sum(Jvec_params.*Jvec_params));

    d = a(:,end-nn_lsizes(end)+1:end) - y;
LGG=true;
    for i=fliplr(2:size(nn_lsizes,2));
        ThetaRows = nn_lsizes(i);
        ThetaCols = nn_lsizes(i-1)+1;

        params_pos -= ThetaRows*ThetaCols;
        a_pos -= nn_lsizes(i)
        a_posUL =  a_pos+nn_lsizes(i)

        Theta = reshape(nn_params(params_pos+1 : params_pos+ThetaRows*ThetaCols), ThetaRows, ThetaCols);

        if LGG,  a, d*Theta(:,2:end).*sigmoidGrad(a(:,a_pos:a_pos+nn_lsizes(i)))    ,LGG=false;end

        
    end
endfunction

