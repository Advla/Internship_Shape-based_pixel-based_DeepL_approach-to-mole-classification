### Pipeline for Fourier-based contour alignment

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
import skfda

from scipy import ndimage #pour fill holes
from scipy.linalg import block_diag
from plot_contour import plot_largest_contour
from preprocessing import pipeline, get_contour

def expend_fourier(X, Y, M=11):
    """
    Expansion en base de Fourier des points X, Y d'un contour échantillonné.
    Parameters:
    X : array_like
        Coordonnées X des points du contour.
    Y : array_like
        Coordonnées Y des points du contour.
    M : int, optional
        Nombre de coefficients de Fourier à calculer, doit être impair. Par défaut, M=11.

    Returns:
    coeffs : array_like (2, M)
        Coefficients réels de la série de Fourier. La première ligne contient les coefficients pour X, la seconde pour Y.
    """
    #Echantillonnage temporel
    t = np.linspace(0, 1, len(X))

    M = M if M % 2 == 1 else M + 1 #correction pour que M soit impair

    X_t = skfda.FDataGrid(X, t)
    Y_t = skfda.FDataGrid(Y, t)
    fourier_basis = skfda.representation.basis.FourierBasis(n_basis=M)
    X_basis = X_t.to_basis(fourier_basis)
    Y_basis = Y_t.to_basis(fourier_basis)
    
    coefs = np.concatenate((X_basis.coefficients, Y_basis.coefficients), axis=0)

    return coefs, fourier_basis

def standardize_contour(coefs):
    """
    Retire la composante de translation et le facteur d'échelle des coefficients de Fourier.
    """
    rho = np.linalg.norm(coefs[:, 1:]) #Norme des coefficients de Fourier (sans le terme constant, qui est la Translation)
    standardized_contour = coefs[:, 1:]/rho

    return standardized_contour

#on résout le problème de Procrustes orthogonal

def procrustes_orthogonal(coefs_contour, coefs_ref):
    """
    Résout le problème de Procrustes orthogonal pour aligner A_contour sur A_ref.
    
    Args:
        coefs_contour: np.ndarray (2, 2K), coefficients du contour à aligner
        coefs_ref: np.ndarray (2, 2K), coefficients du contour de référence

    Returns:
        O: np.ndarray (2, 2), matrice de rotation qui aligne coefs_contour sur coefs_ref, par une multiplication à gauche
        Z: np.ndarray (2, 2K), contour aligné
    """
    
    #Calcul de la matrice S pour la décomposition SVD

    S=coefs_contour @ np.transpose(coefs_ref)

    SVD=np.linalg.svd(S)

    O=SVD[0] @ SVD[2]
    if np.linalg.det(O) < 0: 
        I_sign=np.identity(2)
        I_sign[1, 1]=-1
        O=SVD[0]@I_sign @ SVD[2]

    return O


def gamma_shifted_coeffs(delta, coefs):
    """
    Applique un déphasage t → t + delta aux coefficients de Fourier réels (sans composante constante).

    Args:
        delta: float ∈ [0, 1]
        coefs: np.ndarray (2, 2K), avec [a1, b1, a2, b2, ..., aK, bK] pour x et y

    Returns:
        coefs_new: np.ndarray (2, 2K), coefficients après déphasage
    """
    coefs = np.asarray(coefs)

    #Définition des angles de rotation, K est le nombre d'harmoniques
    K = coefs.shape[1] // 2

    #Construction des matrices de rotation 2x2 pour chaque harmonique k
    #Pour chaque k, on a la matrice [[cos(k*2π*δ), sin(k*2π*δ)], [-sin(k*2π*δ), cos(k*2π*δ)]]
    R_blocks = []
    for k in range(K):
        R_k = np.array([[np.cos(2 * np.pi * k * delta), np.sin(2 * np.pi * k * delta)], 
                         [-np.sin(2 * np.pi * k * delta), np.cos(2 * np.pi * k * delta)]])
        R_blocks.append(R_k)
    
    #Matrice bloc-diagonale
    R = block_diag(*R_blocks)  # shape (2K, 2K)

    #coefs a la forme (2, 2K) -> on applique la transformation pour chaque coordonnée
    x_coeffs = coefs[0]  # shape (2K,)
    y_coeffs = coefs[1]  # shape (2K,)

    #on applique la matrice de rotation bloc-diagonale à chaque coordonnée
    x_new = R @ x_coeffs  # shape (2K,)
    y_new = R @ y_coeffs  # shape (2K,)

    coefs_new = np.vstack((x_new, y_new))  # shape (2, 2K)
    return coefs_new


def align_contours_with_fourier_shift(coefs_contour, coefs_ref, n_points=200, n_deltas=100):
    """
    Aligne un contour reconstruit par Fourier sur un contour de référence en optimisant le décalage delta.

    Args:
        a_x, b_x, a_y, b_y: coefficients de Fourier pour x(t) et y(t)
        A_ref: np.ndarray (2 x N), contour de référence
        n_points: nombre de points pour échantillonnage
        n_deltas: nombre de valeurs de delta à tester

    Returns:
        dict avec delta optimal, rotation, contour aligné, erreur minimale
    """
    coefs_ref = coefs_ref[:, :n_points]  # Troncature si besoin
    best_dist = np.inf
    best_result = {}

    for delta in np.linspace(0, 1, n_deltas, endpoint=False): #grille de deltas
        #Reparamétrisation du contour avec le décalage delta
        B = gamma_shifted_coeffs(delta, coefs_contour)

        #Procrustes rigide puis alignement de la rotation
        O = procrustes_orthogonal(B, coefs_ref)
        aligned = np.transpose(O) @ B
        #on calcule la distance de Frobenius entre le contour aligné et le contour de référence
        dist = np.linalg.norm(aligned - coefs_ref, ord='fro')

        if dist < best_dist:
            best_dist = dist
            best_result = {
                'delta': delta,
                'O': O,
                'aligned_contour': aligned,
                'frobenius_error': dist
            }

    return best_result

def get_rotation_angle(rotation_matrix):
  """
  Extrait l'angle de rotation d'une matrice de rotation 2x2.

  Args:
    rotation_matrix: Une matrice 2x2 numpy.ndarray représentant la rotation.

  Returns:
    L'angle de rotation en radians.
  """
  return np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

def alignment_pipeline(mask, standardized_ref_coefs, n_deltas=100, M=11):
    """
    Pipeline d'alignement de contour basé sur la série de Fourier.
    Parameters:
        mask : np.ndarray
        Masque binaire du contour à aligner.
        standardized_ref_coefs : np.ndarray
        Coefficients de Fourier standardisés du contour de référence. (2, M-1)
        n_deltas : int, optional
        Nombre de valeurs de delta à tester pour l'alignement. Par défaut, n_deltas=100.
        M : int, optional
        Nombre de coefficients de Fourier à utiliser. Par défaut, M=11.
    Returns:
        coefs_aligned : np.ndarray
        Coefficients de Fourier du contour aligné.
        T: np.ndarray
        Coefficients de translation (premier coefficient de Fourier).
        rho: float
        Norme des coefficients sans le terme constant.
        rotation_angle: float
        Angle de rotation appliqué au contour.
        delta: float
        Phase shift appliqué au contour.
        frobenius_error: float
        Erreur de Frobenius entre le contour aligné et le contour de référence.
    """
    #Extraction du contour du masque
    contour = get_contour(mask)
    
    #extraction des coordonnées X, Y
    X, Y = contour[:, 0, 0], contour[:, 0, 1]
    
    #Expansion en série de Fourier
    coefs, fourier_basis = expend_fourier(X, Y, M)
    
    #Standardisation des coefficients
    standardized_coefs = standardize_contour(coefs)

    #Alignement des contours
    result = align_contours_with_fourier_shift(standardized_coefs, standardized_ref_coefs, n_deltas)
    coefs_aligned = result['aligned_contour'].flatten()  # Flatten pour obtenir un vecteur 1D
    T = coefs[:, 0]  # Coefficients de translation (premier coefficient de Fourier)
    rho = np.linalg.norm(coefs[:, 1:])  #Norme des coefficients sans le terme constant
    coefs = np.array([rho, get_rotation_angle(result['O']), result['delta'], result['frobenius_error']])

    return np.concatenate([coefs_aligned, T, coefs], axis=0)
