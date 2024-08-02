# Invoke-WebRequest -Uri https://github.com/microsoft/winget-cli/releases/download/v1.3.2691/Microsoft.DesktopAppInstaller_8wekyb3d8bbwe.msixbundle -OutFile .\MicrosoftDesktopAppInstaller_8wekyb3d8bbwe.msixbundle
winget install Microsoft.VisualStudio.2022.BuildTools Kitware.CMake #make sure to also select desktop developement with C++
# uncomment the first line if you dont have winget installed for some reason

wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.x.zip
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.x.zip
Expand-Archive -Path ./opencv.zip
Expand-Archive -Path ./opencv_contrib.zip
downloading required stuff and extracting it
cd ../

$P_REPO_LOCATION = Get-Location

New-Item -ItemType Directory -Path ./installers/build
cd ./installers/build


$CMAKE_OPTIONS = '-DBUILD_PERF_TESTS:BOOL=OFF -DBUILD_TESTS:BOOL=OFF -DBUILD_DOCS:BOOL=OFF  -DWITH_CUDA:BOOL=OFF -DBUILD_EXAMPLES:BOOL=OFF -DINSTALL_CREATE_DISTRIB=ON'
$CMAKE_GENERATOR_OPTIONS = '-G Visual Studio 17 2022'


Write-host "$CMAKE_OPTIONS"
Write-host "$CMAKE_GENERATOR_OPTIONS"
Write-host "$P_REPO_LOCATION" #for verification purposes

cmake "$CMAKE_GENERATOR_OPTIONS" "$CMAKE_OPTIONS" -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/opencv_contrib-4.x/modules -DCMAKE_INSTALL_PREFIX="$P_REPO_LOCATION/opencv" "../opencv/opencv-4.x"
cmake --build .  --target install --config release
