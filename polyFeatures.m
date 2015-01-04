function X_poly = polyFeatures(X)
X_poly = X;

for i=1:size(X,2)
    for j=i:size(X,2)
        X_poly = [X_poly X(:,i).*X(:,j)];
    end
end

for i=1:size(X,2)
    for j=i:size(X,2)
        for u=j:size(X,2)
            X_poly = [X_poly X(:,i).*X(:,j).*X(:,u)];
        end
    end
end

endfunction

