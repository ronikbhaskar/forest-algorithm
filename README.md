# Forest Algorithm

## Introduction to Forest

I can't remember the last time I played tic-tac-toe, probably because it ended in a draw. Hoping to bring back the excitment and strategy of the game, I created a variation I call "Forest." The name comes from "four-by-four torus tic-tac-toe" being reduced to "fourus t." which sounds like "forest" when you say it out loud. The game is tic-tac-toe played on a 4x4 grid, but rather than playing on a flat plane, the game is played on the surface of a donut—a torus—which offers some interesting opportunities for win cases. It's a blast to play, much more fun than tic-tac-toe, but now I have a new problem. I don't have a good strategy. There are a finite set of board states, so theoretically, we could map out every game and create a perfect strategy.

Theoretically.

## Turning Theory into Practice (My Strategy)

If you were to map out every possible decision on a Forest board, you would have roughly 30 trillion decisions. Once we map out this decision tree once, it'll be done, and we won't have to do it again. The problem arises when you want to map out the tree once. It's just too big to do in any reasonable amount of time. I won't settle for anything less than perfect, so I will create the tree, but I'll have to figure out how to do it faster.

In my view, there are two ways to optimize a procedure. Either you do the same procedure, but faster, or you get clever with your work. With the first option, it's like walking a lap versus sprinting. You use efficient encodings, reuse similar chunks of memory, and perform faster operations, but if you were to describe how the algorithm works at a high level, it would sound the same before and after the optimization, "I go all the way around the track." With the second option, it's like walking a lap versus walking halfway then cutting across the field directly back to the start. You fundamentally change how you approach the problem, whether translating the problem, taking advantage of certain assumptions, or just doing less work.

Given my previous experiences with decision trees, I started with trying to do less work. While the decision tree has 30 trillion branches, the total number of possible board states is less than 4.3 million, which is still a lot, but a substantial reduction. This means the decision tree is repeating states, which makes sense. As an example, consider these two games of tic-tac-toe (not Forest):

```
  |  |        X |  |        X |  |        X |  | X
--------      --------      --------      --------
  |  |    -->   |  |    -->   |O |    -->   |O |  
--------      --------      --------      --------
  |  |          |  |          |  |          |  |   
```

```
  |  |          |  | X        |  | X      X |  | X
--------      --------      --------      --------
  |  |    -->   |  |    -->   |O |    -->   |O |  
--------      --------      --------      --------
  |  |          |  |          |  |          |  |   
```

In both, they reach the same state by the end of the third move, but they started with different decisions, which means they represent separate branches on the decision tree. We could reduce the work we need to do by reusing this state in multiple places on the decision tree. Just like that, we went from the tens of trillions of states to examine to just a few million, but we can do less.

After taking nearly a dozen math classes, I think I will forever have equivalence relations drilled into my head. Essentially, equivalence relations are a formalized method to describe when different objects are equal based on a definition of equality you decide. Since Forest is played on a torus, we can rotate, twist, and mirror the torus, all while preserving the game. As such, transforming the torus in different ways creates equivalent states, and we can let this be our definition of equivalence. To actually generate these equivalent states, I used formal representations of the ways you can manipulate a 4x4 torus and computed 128 transformations. With this, we need to check even fewer states, only around 77 thousand. 

After all this optimization, I started the program and waited to see some progress. For some reason, my program was only calculating a few optimal moves per second. I know my laptop isn't necessarily state-of-the-art, but it should be faster than this. I couldn't reduce the number of states I was checking, so it was time to try the other part of optimization. Just do it faster.

This project was created in Python, as most of my projects are, since it's the language in which I can think the fastest. Translating my thoughts to Python is second nature at this point, but that doesn't mean translating Python to fast machine instructions is particularly quick. Fortunately, I have a few years of lower-level programming experience, so I started going through my program, trying to consider what operations would be slow or where I might be allocating too much memory, and I found one very clear bottleneck. My lookup table, with its fancy equivalence relation, turned the board into a lookup key, and then the key back into a board. When turning the key into a new board, it created a new board, which meant I was doing dozens of memory allocations just to use a single state for a few operations. To fix this, I just reused an old board and overwrote the memory there. 

I made a few similar changes, removing redundant data structures with fancy sorting and simplifying to basic arithmetic operations. All of a sudden, my basic minimax implementation was flying, and the program generated a table with all of the optimal moves in just 22 seconds.

## Reinforcement Learning (Dawson's Strategy)

While I was working on this, Dawson took a different approach. Rather than using massive lookup space reductions, he turned to Reinforcement Learning with Monte-Carlo Tree Search. I'll leave him to give an explanation of his process and how he approached this problem.

## Two Will Enter, Only One Will Leave

Now that we have our working Forest bots, we just need to pit them against each other and see how they compare.

