\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{cours}

\LoadClassWithOptions{article}
\usepackage{cmap}
\usepackage{lmodern}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[british, french]{babel}
\usepackage{texnames}
\usepackage{array}
\usepackage{tikz}
\usepackage{tikzfill}
\usepackage{booktabs}
\usepackage{textcomp}
\usepackage{pdfpages}
\tikzset{>=stealth}
\usepackage{ragged2e}
\usepackage{amsmath}
\usepackage{dsfont}
\usepackage{amsfonts}
\usepackage{mathrsfs}
\usepackage{amsthm}
\usepackage{amssymb}
\usepackage{multirow}
\usepackage{listings}
\usepackage{algorithm}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage{microtype}
\usepackage{algpseudocode}
\usepackage{etoc}
\usepackage{stmaryrd}
\setcounter{secnumdepth}{4}
\usepackage{wrapfig}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage[justification=centering]{caption}

\author{Null}
\title{None}

\usepackage{fancyhdr}
\pagestyle{fancy}

\usepackage{contour}
\usepackage{soul}
\setulcolor{yulm}

\renewcommand{\headrulewidth}{0.5pt}

\fancyhf{}

\fancyhead[LE, RO]{\sffamily\thepage}

\fancyhead[LO]{\rightmark}

\fancyhead[RE]{\leftmark}

\fancypagestyle{plain}{
        \fancyhead{}
        \renewcommand{\headrulewidth}{0pt}
}

\usepackage{emptypage}


\AtBeginDocument{\def\labelitemi{$\bullet$}}
\newtheorem{theorem}{Théorème}[subsection]
\newtheorem{corollary}{Corollaire}[theorem]
\newtheorem{lemma}[theorem]{Lemme}
\newtheorem{remark}{Remarque}[theorem]
\newtheorem{proposition}{Proposition}[subsection]
\newtheorem{definition}{Définition}[subsection]
\newtheorem{example}{Exemple}[subsection]


\renewcommand{\d}{{\, \rm d}}

%Symboles
\newcommand{\Id}{\mathrm{Id}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\Q}{\mathbb{Q}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\C}{\mathbb{C}}
\newcommand{\K}{\mathbb{K}}
\newcommand{\U}{\mathbb{U}}
\renewcommand{\O}{\mathcal{O}}
\newcommand{\A}{\mathcal{A}}
\newcommand{\mL}{\mathcal{L}}
\newcommand{\B}{\mathcal{B}}
\newcommand{\ml}{\mathcal{l}}
\newcommand{\mP}{\mathcal{P}}
\renewcommand{\P}{\mathbb{P}}
\renewcommand{\d}{{\rm \, d}}

\newcommand{\asympt}{\mathop{\sim}}
\newcommand{\lin}{\mathcal{L}}
\newcommand{\M}{\mathcal{M}}
\newcommand{\cont}{\mathcal{C}}
\renewcommand{\phi}{\varphi}
\renewcommand{\epsilon}{\varepsilon}
\renewcommand{\emptyset}{\varnothing}
\renewcommand\qedsymbol{$\blacksquare$}

%Operators
\DeclareMathOperator{\Card}{Card}
\DeclareMathOperator{\Gl}{Gl}
\DeclareMathOperator{\pgcd}{pgcd}
\DeclareMathOperator{\Tr}{Tr}
\DeclareMathOperator{\Aut}{Aut}

\DeclareMathOperator{\rg}{rg}
\DeclareMathOperator{\ch}{ch}
\DeclareMathOperator{\sh}{sh}
\DeclareMathOperator{\sgn}{sgn}
\renewcommand{\th}{\mathop{\rm th}}
\makeatletter
\renewcommand{\ALG@name}{Algorithme}
\newcommand{\algorithmautorefname}{Algorithme}
\makeatother

\algblock{Input}{EndInput}
\algnotext{EndInput}


%Macros
\newcommand{\applic}[4]{\begin{array}[t]{rcl}
#1 & \to & #2 \\
#3 & \mapsto & #4
\end{array}}

\newcommand{\norm}[1]{\left\lVert #1 \right\rVert}
\newcommand{\ninf}[1]{\left\lVert #1 \right\rVert_\infty}
\newcommand{\normop}[1]{\left| \left| \left| #1 \right| \right| \right|}
\newcommand{\tocless}[2]{\bgroup\let\addcontentsline=\nocontentsline#1{#2}\egroup}
\newcommand{\abs}[1]{\left|#1\right|}
\newcommand{\scalar}[1]{\left\langle #1 \right\rangle}
\newcommand{\znz}[1]{\Z/#1\Z}
\newcommand{\mat}[2]{\mathscr{M}_{#1}\left(#2\right)}
\newcommand{\poly}[1]{\mathscr{P}_{#1}}
\newcommand{\astuce}{$(\star)$}
\newcommand{\vect}[2][]{\text{Vect}_{#2}\left(#1\right)}
\renewcommand{\H}{\mathbb{H}}
\newcommand{\transpose}[1]{{^{\mathsf{t}}}#1}
\newcommand{\onen}[1]{\left\llbracket 1, #1\right\rrbracket}
\newcommand{\partialderiv}[2]{\frac{\partial #1}{\partial #2}}


\usetikzlibrary{automata, arrows, calc, matrix, positioning}
\usetikzlibrary{math}
\definecolor{vulm}{HTML}{7d1dd3}
\definecolor{yulm}{HTML}{ffe500}

\hypersetup{
    linkbordercolor=vulm,
    citebordercolor=vulm,
    menubordercolor=yulm
}
\usepackage{geometry}

\geometry{
        top=2.5cm,
        bottom=2cm,
        inner=2cm,
        outer=2cm,
        headsep=10pt,
        headheight=0.5cm,
        footskip=1.4cm,
        columnsep=1cm,
        %showframe,
}

\renewcommand{\titlepage}[4]{
        \thispagestyle{empty}
	\begin{tikzpicture}[overlay,remember picture]
         \end{tikzpicture}
        \begin{tikzpicture}[remember picture, overlay]
		\path[fill tile picture={%
		      \node[inner sep=0pt,outer sep=0pt] {\usebox{\tileone}};
		    }] (current page.south east) rectangle (current page.north west);
		\node [inner sep=0pt] at (current page.center) {#1};
		\node [anchor=center, yshift=4cm, inner sep=1.25cm, rectangle, fill=vulm!50, fill opacity=1, text opacity=1, minimum height=0.2\paperheight, text width=.9\paperwidth, minimum width=\paperwidth] (title) at (current page.center) {#2};
		\node [anchor=north east, rectangle, fill=white, fill opacity=.5, text opacity=1, yshift=-1.4cm, xshift=-.1\paperwidth] at (title.south) {\includegraphics[width=.3\linewidth]{#3}};
		\node [anchor=south, yshift=3cm, inner sep=1.25cm, minimum height=0.2\paperheight, minimum width=\paperwidth, text width=0.9\paperwidth, rectangle, fill=white, fill opacity=.5, text opacity=1] at (current page.south) {#4};
		\node [anchor=north west, rectangle, fill=white, fill opacity=.5, text opacity=1, xshift=+.1\paperwidth, yshift=-1.4cm] at (title.south) {\includegraphics[width=.3\linewidth]{~/Documents/ETUDES/ENS/ens_psl}};
        \end{tikzpicture}
        \cleardoublepage\pagenumbering{arabic}
}

\usepackage{adforn}
\definecolor{stictchbrown}{HTML}{E7D192}

\newsavebox{\tileone}
\sbox{\tileone}{%
  \begin{tikzpicture}[x=1cm,y=1cm]
  \clip (0,0) rectangle (2,2);
  \foreach \x in {0,1} {
    \foreach \y in {0,1} {
       \node[font=\Large,text=vulm!20]
         at (2*\x, 2*\y) {$\beth$};
       \node[font=\Large,text=yulm!80!black!40]
         at (2*\x+1, 2*\y+1) {$\aleph$};
     }
  }
  \begin{scope}[dashed, double, stictchbrown!50]
    \draw plot (\x,\x-1);
    \draw plot (\x,\x+1);
    \draw plot (\x,-\x+1);
    \draw plot (\x,-\x+3);
  \end{scope}
  \end{tikzpicture}%
}

\usepackage[skins]{tcolorbox}
\tcbuselibrary{theorems}
\NewTcbTheorem[auto counter, number within=section]{définition}{Définition}%
 {colback=yulm!10, colframe=vulm!45!black, fonttitle=\bfseries}{def}

\NewTcbTheorem[auto counter, number within=section]{théorème}{Théorème}%
 {colback=yulm!10!white, colframe=vulm!65!black, fonttitle=\bfseries}{thm}

\NewTcbTheorem[auto counter, number within=section]{propositionfr}{Proposition}%
 {colback=yulm!10!white, colframe=vulm!25!black, fonttitle=\bfseries}{propo}

\NewTcbTheorem[auto counter, number within=section]{corollaire}{Corollaire}%
 {colback=yulm!10, colframe=vulm!35!black, fonttitle=\bfseries}{cor}

\NewTcbTheorem[auto counter, number within=section]{lemme}{Lemme}%
 {colback=yulm!10, colframe=vulm!15!black, fonttitle=\bfseries}{lemme}

\NewTcbTheorem[auto counter, number within=section]{remarque}{Remarque}%
 {colback=yulm!10, colframe=vulm!65!black, fonttitle=\bfseries}{rem}
