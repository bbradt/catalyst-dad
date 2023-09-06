import torch
from distributed_auto_differentiation.utils import dprint

def power_iteration_BC(B, C, rank=3, numiterations=1, device='cuda', tol=1e-3, compute_sigma=True):
    """This function is the main workhorse of rank-dAD. The original function was 
            written by Sergey Plis in numpy, then translated into pytorch by Bradley Baker,
            and then modified by Sergey Plis and Bradley Baker. 

        Args: 
            B - The initial left matrix for the power-iteration based factorization
            C - the initial right matrix for the power-iteration based factorization
        Kwargs:
            rank - the maximum effective rank to use
            numiterations - the number of power iterations to perform
            device - the device where tensors will remain
            tol - a float tolerance used for removing "insignificant" eigenvectors
        Returns:
            The "rank" top left and right eigenvectors
    """
    #dprint(rank, tol, compute_sigma, B.shape, C.shape)
    [cm, cn] = C.shape
    if cm > cn:
        CC = torch.mm(C.T, C)
        BCC = torch.mm(B, CC)
    else:
        BCT = torch.mm(B, C.T)
        BCC = torch.mm(BCT, BCT.T)
    #dprint(BCT.shape, BCC.shape)
    #dprint(CC.shape, BCC.shape, C.shape, B.shape, BCC.shape)
    #dprint(torch.linalg.matrix_rank(BCC))

    def zero_result():        
        sigma = torch.tensor(0.0, device=device)
        b_k = torch.zeros(B.shape[0], device=device)
        c_k = torch.zeros(C.shape[0], device=device)
        return {"b": b_k, "c": c_k, "sigma": sigma, "v": b_k}

    def eigenvalue(B, v):
        Bv = torch.mv(B.T, v)
        return torch.sqrt(Bv.dot(torch.mv(CC, Bv)))

    def eigenvalue2(B, v):
        Bv = torch.mv(torch.mm(C, B.T), v)
        return torch.sqrt(Bv.T.dot(Bv))

    def past_values(computed_eigs):
        bb = torch.stack([x['b'] for x in computed_eigs], 0)
        vv = torch.stack([x['v'] for x in computed_eigs], 0)
        return bb, vv

    def iterations(computed_eigs=[], is_sigma=1):
        if not is_sigma: return zero_result()
        # start with one of the columns
        b_k = torch.rand(B.shape[0], device=device)
        if computed_eigs:
            bb, vv = past_values(computed_eigs)
        for _ in range(numiterations):
            adjuster = torch.tensor(0.0, device=device)
            if computed_eigs:
                adjuster = torch.mv(vv.T, torch.mv(bb, b_k))
            # calculate the matrix-by-vector product (BC'CB' - adjusting_matrix)b
            if cm > cn:
                b_k1 = torch.mv(BCC, torch.mv(B.T, b_k)) - adjuster
            else:
                b_k1 = torch.mv(BCC, b_k) - adjuster
            # calculate the norm of b
            b_k1_norm = torch.norm(b_k1)
            # re normalize the vector
            b_k = b_k1 / b_k1_norm
        if compute_sigma:
            if cm > cn:
                sigma = eigenvalue(B, b_k)
            else:
                sigma = eigenvalue2(B, b_k)
            if torch.isnan(sigma): return zero_result()
        else:
            sigma = 1
        if cm > cn:
            c_k = torch.mv(C, torch.mv(B.T, b_k)) / sigma
        else:
            c_k = torch.mv(BCT.T, b_k) / sigma
        if len(computed_eigs) > 1 and torch.norm(b_k - computed_eigs[-1]['b']) / torch.norm(
                computed_eigs[-1]['b']) < tol:
            r = zero_result()
            computed_eigs[-1]['b'] = r['b']
            computed_eigs[-1]['c'] = r['c']
            computed_eigs[-1]['sigma'] = r['sigma']
            return zero_result()
        return {"b": b_k, "c": c_k, "sigma": sigma, "v": sigma * sigma * b_k}

    eigs = [{"sigma": torch.tensor(1.0, device=device)}]
    for i in range(rank):
        eigs += [iterations(computed_eigs=eigs[1:], is_sigma=eigs[-1]["sigma"])]
        if eigs[-1]["sigma"] == 0.0:
            break
    eigs = eigs[1:-1]
    return (
        torch.stack([x["sigma"] * x["b"] for x in eigs], 1),
        torch.stack([x["c"] for x in eigs], 1))
