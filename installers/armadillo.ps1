#winget install Microsoft.VisualStudio.2022.BuildTools Kitware.CMake Git.Git

git clone https://gitlab.com/conradsnicta/armadillo-code.git

cd ../

$P_REPO_LOCATION = Get-Location

cd ./installers

cd armadillo-code

New-Item -ItemType Directory -Path ./build
cd ./build

$CMAKE_GENERATOR_OPTIONS = '-G Visual Studio 17 2022'

cmake "$CMAKE_GENERATOR_OPTIONS" -DCMAKE_INSTALL_PREFIX="$P_REPO_LOCATION/armadillo-code" ..
cmake --build .  --target install --config release
