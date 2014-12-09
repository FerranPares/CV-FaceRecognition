function PCAImagesDim = apply_pca(images, dim)
    % APPLY_PCA 
    % PCAImagesDim It returns the matrix with reduced attributes.
    % Each column is an image
    % 

    %% PCA
    % Use princomp.m to compute:
    % 5. To complete:
    % [PCACoefficients, PCAImages, PCAValues] = ...
    [PCACoefficients, PCAImages, PCAValues] = princomp(images);

    %% Show the 30 first eigenfaces
    % 6. To complete:
    % show_eigenfaces(...);
    show_eigenfaces(PCACoefficients);

    %% Plot the explained variance using 100 dimensions
    % 7. To complete:
    p = 1:100;
    figure;
    plot(p,PCAValues(1:100));

    %% Keep the first 'dim' dimensions where dim is given or computed as the
    %% dimensions necessary to preserve 90% of the data variance.
    if dim>0
        PCAImagesDim = PCAImages(:,1:dim);
    else
        % Compute the number of dimensions necessary to preserve 95% of the data variance.
        % 8. To complete:
        Tvar = sum(PCAValues(1:100));
        PoV = 0;
        while PoV<0.95
            dim = dim + 1;
            PoV = sum(PCAValues(1:dim))/Tvar;
        end
        PCAImagesDim = PCAImages(:,1:dim);
    end
end