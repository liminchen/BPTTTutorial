% This program tests the BPTT process we manually developed for GRU.
% We calculate the gradients of GRU parameters with chain rule, and then
% compare them to the numerical gradients to check whether our chain rule
% derivation is correct.

% Here, we provided 2 versions of BPTT, backward_direct() and backward(). 
% The former one is the direct idea to calculate gradient within each step 
% and add them up (O(sentence_size^2) time). The latter one is optimized to
% calculate the contribution of each step to the overall gradient, which is 
% only O(sentence_size) time.

% This is very helpful for people who wants to implement GRU in Caffe since
% Caffe didn't support auto-differentiation. This is also very helpful for
% the people who wants to know the details about Backpropagation Through
% Time algorithm in the Reccurent Neural Networks (such as GRU and LSTM)
% and also get a sense on how auto-differentiation is possible.

% NOTE: We didn't involve SGD training here. With SGD training, this
% program would become a complete implementation of GRU which can be
% trained with sequence data. However, since this is only a CPU serial
% Matlab version of GRU, applying it on large datasets will be dramatically
% slow.

% by Minchen Li, at The University of British Columbia. 2016-04-21

function testBPTT_GRU
    % set GRU and data scale
    vocabulary_size = 64;
    iMem_size = 4;
    sentence_size = 20; % number of words in a sentence 
                        %(including start and end symbol)
                        % since we will only use one sentence for training,
                        % this is also the total steps during training.

    [x y] = getTrainingData(vocabulary_size, sentence_size);

    % initialize parameters:
    % multiplier for input x_t of intermediate variables
    U_z = rand(iMem_size, vocabulary_size);
    U_r = rand(iMem_size, vocabulary_size);
    U_c = rand(iMem_size, vocabulary_size);
    % multiplier for pervious s of intermediate variables
    W_z = rand(iMem_size, iMem_size);
    W_r = rand(iMem_size, iMem_size);
    W_c = rand(iMem_size, iMem_size);
    % bias terms of intermediate variables
    b_z = rand(iMem_size, 1);
    b_r = rand(iMem_size, 1);
    b_c = rand(iMem_size, 1);
    % decoder for generating output
    V = rand(vocabulary_size, iMem_size);
    b_V = rand(vocabulary_size, 1); % bias of decoder
    % previous s of step 1
    s_0 = rand(iMem_size, 1);

    % calculate and check gradient
    tic
    [dV, db_V, dU_z, dU_r, dU_c, dW_z, dW_r, dW_c, db_z, db_r, db_c, ds_0] = ...
        backward_direct(x, y, U_z, U_r, U_c, W_z, W_r, W_c, b_z, b_r, b_c, V, b_V, s_0);
    toc
    tic
    checkGradient_GRU(x, y, U_z, U_r, U_c, W_z, W_r, W_c, b_z, b_r, b_c, V, b_V, s_0, ...
        dV, db_V, dU_z, dU_r, dU_c, dW_z, dW_r, dW_c, db_z, db_r, db_c, ds_0);
    toc
    
    tic
    [dV, db_V, dU_z, dU_r, dU_c, dW_z, dW_r, dW_c, db_z, db_r, db_c, ds_0] = ...
        backward(x, y, U_z, U_r, U_c, W_z, W_r, W_c, b_z, b_r, b_c, V, b_V, s_0);
    toc
    tic
    checkGradient_GRU(x, y, U_z, U_r, U_c, W_z, W_r, W_c, b_z, b_r, b_c, V, b_V, s_0, ...
        dV, db_V, dU_z, dU_r, dU_c, dW_z, dW_r, dW_c, db_z, db_r, db_c, ds_0);
    toc
end

% Forward propagate calculate s, y_hat, loss and intermediate variables for each step
function [s, y_hat, L, z, r, c] = forward(x, y, ...
    U_z, U_r, U_c, W_z, W_r, W_c, b_z, b_r, b_c, V, b_V, s_0)
    % count sizes
    [vocabulary_size, sentence_size] = size(x);
    iMem_size = size(V, 2);
    
    % initialize results
    s = zeros(iMem_size, sentence_size);
    y_hat = zeros(vocabulary_size, sentence_size);
    L = zeros(sentence_size, 1);
    z = zeros(iMem_size, sentence_size);
    r = zeros(iMem_size, sentence_size);
    c = zeros(iMem_size, sentence_size);
    
    % calculate result for step 1 since s_0 is not in s
    z(:,1) = sigmoid(U_z*x(:,1) + W_z*s_0 + b_z);
    r(:,1) = sigmoid(U_r*x(:,1) + W_r*s_0 + b_r);
    c(:,1) = tanh(U_c*x(:,1) + W_c*(s_0.*r(:,1)) + b_c);
    s(:,1) = (1-z(:,1)).*c(:,1) + z(:,1).*s_0;
    y_hat(:,1) = softmax(V*s(:,1) + b_V);
    L(1) = sum(-y(:,1).*log(y_hat(:,1)));
    % calculate results for step 2 - sentence_size similarly
    for wordI = 2:sentence_size
        z(:,wordI) = sigmoid(U_z*x(:,wordI) + W_z*s(:,wordI-1) + b_z);
        r(:,wordI) = sigmoid(U_r*x(:,wordI) + W_r*s(:,wordI-1) + b_r);
        c(:,wordI) = tanh(U_c*x(:,wordI) + W_c*(s(:,wordI-1).*r(:,wordI)) + b_c);
        s(:,wordI) = (1-z(:,wordI)).*c(:,wordI) + z(:,wordI).*s(:,wordI-1);
        y_hat(:,wordI) = softmax(V*s(:,wordI) + b_V);
        L(wordI) = sum(-y(:,wordI).*log(y_hat(:,wordI)));
    end
end

% Backward propagate to calculate gradient using chain rule
% (O(sentence_size) time)
function [dV, db_V, dU_z, dU_r, dU_c, dW_z, dW_r, dW_c, db_z, db_r, db_c, ds_0] = ...
    backward(x, y, U_z, U_r, U_c, W_z, W_r, W_c, b_z, b_r, b_c, V, b_V, s_0)
    % forward propagate to get the intermediate and output results
    [s, y_hat, L, z, r, c] = forward(x, y, U_z, U_r, U_c, W_z, W_r, W_c, ...
        b_z, b_r, b_c, V, b_V, s_0);
    % count sentence size
    [~, sentence_size] = size(x);
    
    % calculate gradient using chain rule
    delta_y = y_hat - y;
    db_V = sum(delta_y, 2);
    
    dV = zeros(size(V));
    for wordI = 1:sentence_size
        dV = dV + delta_y(:,wordI)*s(:,wordI)';
    end
    
    ds_0 = zeros(size(s_0));
    dU_c = zeros(size(U_c));
    dU_r = zeros(size(U_r));
    dU_z = zeros(size(U_z));
    dW_c = zeros(size(W_c));
    dW_r = zeros(size(W_r));
    dW_z = zeros(size(W_z));
    db_z = zeros(size(b_z));
    db_r = zeros(size(b_r));
    db_c = zeros(size(b_c));
    ds_single = V'*delta_y;
    % calculate the derivative contribution of each step and add them up
    ds_cur = zeros(size(ds_single,1), 1);
    for wordJ = sentence_size:-1:2
        ds_cur = ds_cur + ds_single(:,wordJ);
        ds_cur_bk = ds_cur;

        dtanhInput = (ds_cur.*(1-z(:,wordJ)).*(1-c(:,wordJ).*c(:,wordJ)));
        db_c = db_c + dtanhInput;
        dU_c = dU_c + dtanhInput*x(:,wordJ)'; %could be accelerated by avoiding add 0
        dW_c = dW_c + dtanhInput*(s(:,wordJ-1).*r(:,wordJ))';
        dsr = W_c'*dtanhInput;
        ds_cur = dsr.*r(:,wordJ);
        dsigInput_r = dsr.*s(:,wordJ-1).*r(:,wordJ).*(1-r(:,wordJ));
        db_r = db_r + dsigInput_r;
        dU_r = dU_r + dsigInput_r*x(:,wordJ)'; %could be accelerated by avoiding add 0
        dW_r = dW_r + dsigInput_r*s(:,wordJ-1)';
        ds_cur = ds_cur + W_r'*dsigInput_r;

        ds_cur = ds_cur + ds_cur_bk.*z(:,wordJ);
        dz = ds_cur_bk.*(s(:,wordJ-1)-c(:,wordJ));
        dsigInput_z = dz.*z(:,wordJ).*(1-z(:,wordJ));
        db_z = db_z + dsigInput_z;
        dU_z = dU_z + dsigInput_z*x(:,wordJ)'; %could be accelerated by avoiding add 0
        dW_z = dW_z + dsigInput_z*s(:,wordJ-1)';
        ds_cur = ds_cur + W_z'*dsigInput_z;
    end
    
    % s_1
    ds_cur = ds_cur + ds_single(:,1);
    
    dtanhInput = (ds_cur.*(1-z(:,1)).*(1-c(:,1).*c(:,1)));
    db_c = db_c + dtanhInput;
    dU_c = dU_c + dtanhInput*x(:,1)'; %could be accelerated by avoiding add 0
    dW_c = dW_c + dtanhInput*(s_0.*r(:,1))';
    dsr = W_c'*dtanhInput;
    ds_0 = ds_0 + dsr.*r(:,1);
    dsigInput_r = dsr.*s_0.*r(:,1).*(1-r(:,1));
    db_r = db_r + dsigInput_r;
    dU_r = dU_r + dsigInput_r*x(:,1)'; %could be accelerated by avoiding add 0
    dW_r = dW_r + dsigInput_r*s_0';
    ds_0 = ds_0 + W_r'*dsigInput_r;

    ds_0 = ds_0 + ds_cur.*z(:,1);
    dz = ds_cur.*(s_0-c(:,1));
    dsigInput_z = dz.*z(:,1).*(1-z(:,1));
    db_z = db_z + dsigInput_z;
    dU_z = dU_z + dsigInput_z*x(:,1)'; %could be accelerated by avoiding add 0
    dW_z = dW_z + dsigInput_z*s_0';
    ds_0 = ds_0 + W_z'*dsigInput_z;
end

% A more direct view of backward propagate to calculate gradient using 
% chain rule. (O(sentence_size^2) time)
% Instead of calculating how much contribution of derivative each step has,
% here we calculate the gradient within every step.
function [dV, db_V, dU_z, dU_r, dU_c, dW_z, dW_r, dW_c, db_z, db_r, db_c, ds_0] = ...
    backward_direct(x, y, U_z, U_r, U_c, W_z, W_r, W_c, b_z, b_r, b_c, V, b_V, s_0)
    % forward propagate to get the intermediate and output results
    [s, y_hat, L, z, r, c] = forward(x, y, U_z, U_r, U_c, W_z, W_r, W_c, ...
        b_z, b_r, b_c, V, b_V, s_0);
    % count sentence size
    [~, sentence_size] = size(x);
    
    % calculate gradient using chain rule
    delta_y = y_hat - y;
    db_V = sum(delta_y, 2);
    
    dV = zeros(size(V));
    for wordI = 1:sentence_size
        dV = dV + delta_y(:,wordI)*s(:,wordI)';
    end
    
    ds_0 = zeros(size(s_0));
    dU_c = zeros(size(U_c));
    dU_r = zeros(size(U_r));
    dU_z = zeros(size(U_z));
    dW_c = zeros(size(W_c));
    dW_r = zeros(size(W_r));
    dW_z = zeros(size(W_z));
    db_z = zeros(size(b_z));
    db_r = zeros(size(b_r));
    db_c = zeros(size(b_c));
    ds_single = V'*delta_y;
    % calculate the derivatives in each step and add them up
    for wordI = 1:sentence_size
        ds_cur = ds_single(:,wordI);
        % since in each step t, the derivatives depends on s_0 - s_t,
        % we need to trace back from t ot 0 each time
        for wordJ = wordI:-1:2
            ds_cur_bk = ds_cur;
            
            dtanhInput = (ds_cur.*(1-z(:,wordJ)).*(1-c(:,wordJ).*c(:,wordJ)));
            db_c = db_c + dtanhInput;
            dU_c = dU_c + dtanhInput*x(:,wordJ)'; %could be accelerated by avoiding add 0
            dW_c = dW_c + dtanhInput*(s(:,wordJ-1).*r(:,wordJ))';
            dsr = W_c'*dtanhInput;
            ds_cur = dsr.*r(:,wordJ);
            dsigInput_r = dsr.*s(:,wordJ-1).*r(:,wordJ).*(1-r(:,wordJ));
            db_r = db_r + dsigInput_r;
            dU_r = dU_r + dsigInput_r*x(:,wordJ)'; %could be accelerated by avoiding add 0
            dW_r = dW_r + dsigInput_r*s(:,wordJ-1)';
            ds_cur = ds_cur + W_r'*dsigInput_r;
            
            ds_cur = ds_cur + ds_cur_bk.*z(:,wordJ);
            dz = ds_cur_bk.*(s(:,wordJ-1)-c(:,wordJ));
            dsigInput_z = dz.*z(:,wordJ).*(1-z(:,wordJ));
            db_z = db_z + dsigInput_z;
            dU_z = dU_z + dsigInput_z*x(:,wordJ)'; %could be accelerated by avoiding add 0
            dW_z = dW_z + dsigInput_z*s(:,wordJ-1)';
            ds_cur = ds_cur + W_z'*dsigInput_z;
        end
        
        % s_1
        dtanhInput = (ds_cur.*(1-z(:,1)).*(1-c(:,1).*c(:,1)));
        db_c = db_c + dtanhInput;
        dU_c = dU_c + dtanhInput*x(:,1)'; %could be accelerated by avoiding add 0
        dW_c = dW_c + dtanhInput*(s_0.*r(:,1))';
        dsr = W_c'*dtanhInput;
        ds_0 = ds_0 + dsr.*r(:,1);
        dsigInput_r = dsr.*s_0.*r(:,1).*(1-r(:,1));
        db_r = db_r + dsigInput_r;
        dU_r = dU_r + dsigInput_r*x(:,1)'; %could be accelerated by avoiding add 0
        dW_r = dW_r + dsigInput_r*s_0';
        ds_0 = ds_0 + W_r'*dsigInput_r;

        ds_0 = ds_0 + ds_cur.*z(:,1);
        dz = ds_cur.*(s_0-c(:,1));
        dsigInput_z = dz.*z(:,1).*(1-z(:,1));
        db_z = db_z + dsigInput_z;
        dU_z = dU_z + dsigInput_z*x(:,1)'; %could be accelerated by avoiding add 0
        dW_z = dW_z + dsigInput_z*s_0';
        ds_0 = ds_0 + W_z'*dsigInput_z;
    end
end

% Sigmoid function for neural network
function val = sigmoid(x)
    val = sigmf(x,[1 0]);
end

% Fake a training data set: generate only one sentence for training.
%!!! Only for testing. Needs to be changed to read in training data from files.
function [x_t, y_t] = getTrainingData(vocabulary_size, sentence_size)
    assert(vocabulary_size > 2); % for start and end of sentence symbol
    assert(sentence_size > 0);
    
    % define start and end of sentence in the vocabulary
    SENTENCE_START = zeros(vocabulary_size, 1);
    SENTENCE_START(1) = 1;
    SENTENCE_END = zeros(vocabulary_size, 1);
    SENTENCE_END(2) = 1;
    
    % generate sentence:
    x_t = zeros(vocabulary_size, sentence_size-1); % leave one slot for SENTENCE_START
    for wordI = 1:sentence_size-1
        % generate a random word excludes start and end symbol
        x_t(randi(vocabulary_size-2,1,1)+2, wordI) = 1;
    end
    y_t = [x_t, SENTENCE_END];   % training output
    x_t = [SENTENCE_START, x_t]; % training input
end

% Use numerical differentiation to approximate the gradient of each
% parameter and calculate the difference between these numerical results
% and our results calculated by applying chain rule.
function checkGradient_GRU(x, y, U_z, U_r, U_c, W_z, W_r, W_c, b_z, b_r, b_c, V, b_V, s_0, ...
    dV, db_V, dU_z, dU_r, dU_c, dW_z, dW_r, dW_c, db_z, db_r, db_c, ds_0)
    % Here we use the centre difference formula:
    %   df(x)/dx = (f(x+h)-f(x-h)) / (2h)
    % It is a second order accurate method with error bounded by O(h^2)
    
    h = 1e-5; 
    % NOTE: h couldn't be too large or too small since large h will
    % introduce bigger truncation error and small h will introduce bigger
    % roundoff error.
    
    dV_numerical = zeros(size(dV));
    % Calculate partial derivative element by element
    for rowI = 1:size(dV_numerical,1)
        for colI = 1:size(dV_numerical,2)
            V_plus = V;
            V_plus(rowI,colI) = V_plus(rowI,colI) + h;
            V_minus = V;
            V_minus(rowI,colI) = V_minus(rowI,colI) - h;
            [~, ~, L_plus] = forward(x, y, ...
                U_z, U_r, U_c, W_z, W_r, W_c, b_z, b_r, b_c, V_plus, b_V, s_0);
            [~, ~, L_minus] = forward(x, y, ...
                U_z, U_r, U_c, W_z, W_r, W_c, b_z, b_r, b_c, V_minus, b_V, s_0);
            dV_numerical(rowI,colI) = (sum(L_plus) - sum(L_minus)) / 2 / h;
        end
    end
    display(sum(sum(abs(dV_numerical-dV)./(abs(dV_numerical)+h))), ...
        'dV relative error'); % prevent dividing by 0 by adding h
    
    dU_c_numerical = zeros(size(dU_c));
    for rowI = 1:size(dU_c_numerical,1)
        for colI = 1:size(dU_c_numerical,2)
            U_c_plus = U_c;
            U_c_plus(rowI,colI) = U_c_plus(rowI,colI) + h;
            U_c_minus = U_c;
            U_c_minus(rowI,colI) = U_c_minus(rowI,colI) - h;
            [~, ~, L_plus] = forward(x, y, ...
                U_z, U_r, U_c_plus, W_z, W_r, W_c, b_z, b_r, b_c, V, b_V, s_0);
            [~, ~, L_minus] = forward(x, y, ...
                U_z, U_r, U_c_minus, W_z, W_r, W_c, b_z, b_r, b_c, V, b_V, s_0);
            dU_c_numerical(rowI,colI) = (sum(L_plus) - sum(L_minus)) / 2 / h;
        end
    end
    display(sum(sum(abs(dU_c_numerical-dU_c)./(abs(dU_c_numerical)+h))), ...
        'dU_c relative error');
    
    dW_c_numerical = zeros(size(dW_c));
    for rowI = 1:size(dW_c_numerical,1)
        for colI = 1:size(dW_c_numerical,2)
            W_c_plus = W_c;
            W_c_plus(rowI,colI) = W_c_plus(rowI,colI) + h;
            W_c_minus = W_c;
            W_c_minus(rowI,colI) = W_c_minus(rowI,colI) - h;
            [~, ~, L_plus] = forward(x, y, ...
                U_z, U_r, U_c, W_z, W_r, W_c_plus, b_z, b_r, b_c, V, b_V, s_0);
            [~, ~, L_minus] = forward(x, y, ...
                U_z, U_r, U_c, W_z, W_r, W_c_minus, b_z, b_r, b_c, V, b_V, s_0);
            dW_c_numerical(rowI,colI) = (sum(L_plus) - sum(L_minus)) / 2 / h;
        end
    end
    display(sum(sum(abs(dW_c_numerical-dW_c)./(abs(dW_c_numerical)+h))), ...
        'dW_c relative error');
    
    dU_r_numerical = zeros(size(dU_r));
    for rowI = 1:size(dU_r_numerical,1)
        for colI = 1:size(dU_r_numerical,2)
            U_r_plus = U_r;
            U_r_plus(rowI,colI) = U_r_plus(rowI,colI) + h;
            U_r_minus = U_r;
            U_r_minus(rowI,colI) = U_r_minus(rowI,colI) - h;
            [~, ~, L_plus] = forward(x, y, ...
                U_z, U_r_plus, U_c, W_z, W_r, W_c, b_z, b_r, b_c, V, b_V, s_0);
            [~, ~, L_minus] = forward(x, y, ...
                U_z, U_r_minus, U_c, W_z, W_r, W_c, b_z, b_r, b_c, V, b_V, s_0);
            dU_r_numerical(rowI,colI) = (sum(L_plus) - sum(L_minus)) / 2 / h;
        end
    end
    display(sum(sum(abs(dU_r_numerical-dU_r)./(abs(dU_r_numerical)+h))), ...
        'dU_r relative error');
    
    dW_r_numerical = zeros(size(dW_r));
    for rowI = 1:size(dW_r_numerical,1)
        for colI = 1:size(dW_r_numerical,2)
            W_r_plus = W_r;
            W_r_plus(rowI,colI) = W_r_plus(rowI,colI) + h;
            W_r_minus = W_r;
            W_r_minus(rowI,colI) = W_r_minus(rowI,colI) - h;
            [~, ~, L_plus] = forward(x, y, ...
                U_z, U_r, U_c, W_z, W_r_plus, W_c, b_z, b_r, b_c, V, b_V, s_0);
            [~, ~, L_minus] = forward(x, y, ...
                U_z, U_r, U_c, W_z, W_r_minus, W_c, b_z, b_r, b_c, V, b_V, s_0);
            dW_r_numerical(rowI,colI) = (sum(L_plus) - sum(L_minus)) / 2 / h;
        end
    end
    display(sum(sum(abs(dW_r_numerical-dW_r)./(abs(dW_r_numerical)+h))), ...
        'dW_r relative error');
    
    dU_z_numerical = zeros(size(dU_z));
    for rowI = 1:size(dU_z_numerical,1)
        for colI = 1:size(dU_z_numerical,2)
            U_z_plus = U_z;
            U_z_plus(rowI,colI) = U_z_plus(rowI,colI) + h;
            U_z_minus = U_z;
            U_z_minus(rowI,colI) = U_z_minus(rowI,colI) - h;
            [~, ~, L_plus] = forward(x, y, ...
                U_z_plus, U_r, U_c, W_z, W_r, W_c, b_z, b_r, b_c, V, b_V, s_0);
            [~, ~, L_minus] = forward(x, y, ...
                U_z_minus, U_r, U_c, W_z, W_r, W_c, b_z, b_r, b_c, V, b_V, s_0);
            dU_z_numerical(rowI,colI) = (sum(L_plus) - sum(L_minus)) / 2 / h;
        end
    end
    display(sum(sum(abs(dU_z_numerical-dU_z)./(abs(dU_z_numerical)+h))), ...
        'dU_z relative error');
    
    dW_z_numerical = zeros(size(dW_z));
    for rowI = 1:size(dW_z_numerical,1)
        for colI = 1:size(dW_z_numerical,2)
            W_z_plus = W_z;
            W_z_plus(rowI,colI) = W_z_plus(rowI,colI) + h;
            W_z_minus = W_z;
            W_z_minus(rowI,colI) = W_z_minus(rowI,colI) - h;
            [~, ~, L_plus] = forward(x, y, ...
                U_z, U_r, U_c, W_z_plus, W_r, W_c, b_z, b_r, b_c, V, b_V, s_0);
            [~, ~, L_minus] = forward(x, y, ...
                U_z, U_r, U_c, W_z_minus, W_r, W_c, b_z, b_r, b_c, V, b_V, s_0);
            dW_z_numerical(rowI,colI) = (sum(L_plus) - sum(L_minus)) / 2 / h;
        end
    end
    display(sum(sum(abs(dW_z_numerical-dW_z)./(abs(dW_z_numerical)+h))), ...
        'dW_z relative error');
    
    db_z_numerical = zeros(size(db_z));
    for i = 1:length(db_z_numerical)
        b_z_plus = b_z;
        b_z_plus(i) = b_z_plus(i) + h;
        b_z_minus = b_z;
        b_z_minus(i) = b_z_minus(i) - h;
        [~, ~, L_plus] = forward(x, y, ...
            U_z, U_r, U_c, W_z, W_r, W_c, b_z_plus, b_r, b_c, V, b_V, s_0);
        [~, ~, L_minus] = forward(x, y, ...
            U_z, U_r, U_c, W_z, W_r, W_c, b_z_minus, b_r, b_c, V, b_V, s_0);
        db_z_numerical(i) = (sum(L_plus) - sum(L_minus)) / 2 / h;
    end
    display(sum(abs(db_z_numerical-db_z)./(abs(db_z_numerical)+h)), ...
        'db_z relative error');
    
    db_r_numerical = zeros(size(db_r));
    for i = 1:length(db_r_numerical)
        b_r_plus = b_r;
        b_r_plus(i) = b_r_plus(i) + h;
        b_r_minus = b_r;
        b_r_minus(i) = b_r_minus(i) - h;
        [~, ~, L_plus] = forward(x, y, ...
            U_z, U_r, U_c, W_z, W_r, W_c, b_z, b_r_plus, b_c, V, b_V, s_0);
        [~, ~, L_minus] = forward(x, y, ...
            U_z, U_r, U_c, W_z, W_r, W_c, b_z, b_r_minus, b_c, V, b_V, s_0);
        db_r_numerical(i) = (sum(L_plus) - sum(L_minus)) / 2 / h;
    end
    display(sum(abs(db_r_numerical-db_r)./(abs(db_r_numerical)+h)), ...
        'db_r relative error');
    
    db_c_numerical = zeros(size(db_c));
    for i = 1:length(db_c_numerical)
        b_c_plus = b_c;
        b_c_plus(i) = b_c_plus(i) + h;
        b_c_minus = b_c;
        b_c_minus(i) = b_c_minus(i) - h;
        [~, ~, L_plus] = forward(x, y, ...
            U_z, U_r, U_c, W_z, W_r, W_c, b_z, b_r, b_c_plus, V, b_V, s_0);
        [~, ~, L_minus] = forward(x, y, ...
            U_z, U_r, U_c, W_z, W_r, W_c, b_z, b_r, b_c_minus, V, b_V, s_0);
        db_c_numerical(i) = (sum(L_plus) - sum(L_minus)) / 2 / h;
    end
    display(sum(abs(db_c_numerical-db_c)./(abs(db_c_numerical)+h)), ...
        'db_c relative error');
    
    db_V_numerical = zeros(size(db_V));
    for i = 1:length(db_V_numerical)
        b_V_plus = b_V;
        b_V_plus(i) = b_V_plus(i) + h;
        b_V_minus = b_V;
        b_V_minus(i) = b_V_minus(i) - h;
        [~, ~, L_plus] = forward(x, y, ...
            U_z, U_r, U_c, W_z, W_r, W_c, b_z, b_r, b_c, V, b_V_plus, s_0);
        [~, ~, L_minus] = forward(x, y, ...
            U_z, U_r, U_c, W_z, W_r, W_c, b_z, b_r, b_c, V, b_V_minus, s_0);
        db_V_numerical(i) = (sum(L_plus) - sum(L_minus)) / 2 / h;
    end
    display(sum(abs(db_V_numerical-db_V)./(abs(db_V_numerical)+h)), ...
        'db_V relative error');
    
    ds_0_numerical = zeros(size(ds_0));
    for i = 1:length(ds_0_numerical)
        s_0_plus = s_0;
        s_0_plus(i) = s_0_plus(i) + h;
        s_0_minus = s_0;
        s_0_minus(i) = s_0_minus(i) - h;
        [~, ~, L_plus] = forward(x, y, ...
            U_z, U_r, U_c, W_z, W_r, W_c, b_z, b_r, b_c, V, b_V, s_0_plus);
        [~, ~, L_minus] = forward(x, y, ...
            U_z, U_r, U_c, W_z, W_r, W_c, b_z, b_r, b_c, V, b_V, s_0_minus);
        ds_0_numerical(i) = (sum(L_plus) - sum(L_minus)) / 2 / h;
    end
    display(sum(abs(ds_0_numerical-ds_0)./(abs(ds_0_numerical)+h)), ...
        'ds_0 relative error');
end