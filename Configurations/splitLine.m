function words = splitLine(text)
    % Get spaces positions
    spaces = find(text == ' ');
    
    words = {};
    % Loop on all spaces positions and insert characters just before and
    % after
    for i = 1 : length(spaces)
        if(i == 1)
            words = [words; text(1 : spaces(i) - 1)];
        elseif (i == length(spaces))
            words = [words; text(spaces(i) + 1 : end)];
        else
            words = [words; text(spaces(i) + 1 : spaces(i+1) - 1)];
        end
    end
end