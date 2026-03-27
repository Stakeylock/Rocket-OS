# Vagrantfile for Rocket-OS Simulation Environment
Vagrant.configure("2") do |config|
  # Use an Ubuntu 22.04 LTS box for stability with scientific python libraries
  config.vm.box = "ubuntu/jammy64"
  
  # Allocate adequate resources for Monte Carlo simulations
  config.vm.provider "virtualbox" do |vb|
    vb.memory = "4096"
    vb.cpus = 4
    vb.name = "rocket_ai_os_sim"
  end

  # Mount the current directory into the VM
  config.vm.synced_folder ".", "/vagrant"

  # Provision the VM with necessary libraries
  config.vm.provision "shell", inline: <<-SHELL
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    apt-get install -y python3-pip python3-dev build-essential git
    
    # Mathematical and optimization libraries required for GNC and Simplex
    # Using specific constraints to ensure stability
    pip3 install numpy scipy pandas seaborn cvxpy tqdm
    
    # Intentionally linking module to system path for easy execution
    pip3 install -e /vagrant
    
    echo "Rocket-OS Simulation Environment is ready!"
    echo "SSH into the machine using 'vagrant ssh', then 'cd /vagrant'."
  SHELL
end