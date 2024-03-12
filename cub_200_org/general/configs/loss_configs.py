recon_coeff = 1
ce_coeff = 1
kl_coeff = 1
l1_coeff = 1
clst_coeff = 1
sep_coeff = 1

def set_loss_coeffs(params):
    global recon_coeff
    global ce_coeff
    global kl_coeff
    global l1_coeff
    global clst_coeff
    global sep_coeff

    recon_coeff = params["recon_coeff"]
    ce_coeff = params["ce_coeff"]
    kl_coeff = params["kl_coeff"]
    li_coeff = params["l1_coeff"]
    clst_coeff = params["clst_coeff"]
    sep_coeff = params["sep_coeff"]