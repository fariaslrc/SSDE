function [Population, Samples, winning_weights]  = Operator(Problem, Population, W, winning_weights)    

    % Generate all candidate offsprings
    MatingPool = TournamentSelection(2,Problem.N,sum(max(0,Population.cons),2))';
    CandidateDec = Population.decs;
    OffspringDec  = OperatorDE(CandidateDec(MatingPool,:), CandidateDec(randi(Problem.N,1,Problem.N),:) , CandidateDec(randi(Problem.N,1,Problem.N),:));    

	% Normalize offspring for SOM mapping
	Normalized_OffspringDec = rescale(OffspringDec,'InputMin',Problem.lower,'InputMax',Problem.upper);
	
	% Distance between each solution to a neuron
    Distance = pdist2(Normalized_OffspringDec,W(:,1:Problem.D)); 
    [~,rank] = sort(Distance,2);    
    
	% Estimate the offspring objective values using SOM
	Offspring_Labels = W(rank(:,1),Problem.D+1:Problem.D+Problem.M);

	%% Assign one to winning nodes
    winning_weights(rank(:,1)) = true;	
	
	%% Selection of survivors by ranking
	objs = [Population.objs;Offspring_Labels];
	cons = [Population.cons;Problem.CalCon(OffspringDec,Offspring_Labels)];
    % cons = [Population.cons;Problem.CalCon(OffspringDec)];
    
	%% Non-dominated sorting
    [FrontNo,MaxFNo] = NDSort(objs,cons,Problem.N);
    Next = FrontNo < MaxFNo;
    
    %% Calculate the crowding distance of each solution
    CrowdDis = CrowdingDistance(objs,FrontNo);
    
    %% Select the solutions in the last front based on their crowding distances
    Last     = find(FrontNo==MaxFNo);
    [~,Rank] = sort(CrowdDis(Last),'descend');
    Next(Last(Rank(1:Problem.N-sum(Next)))) = true;
    
    %% Population for next generation
    out = Next(1:Problem.N);
    in = Next(Problem.N+1:end);

	%% Selection
    if sum(in) >= 1
        Offspring = SOLUTION(OffspringDec(in,:));
        Samples = Offspring;
        Population(~out) = Offspring;
    else
        Samples = Population;
    end
end